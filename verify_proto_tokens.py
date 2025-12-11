import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torchvision.utils import save_image
import numpy as np

# --- LlamaGen/Autoregressive 框架导入 ---
# 注意: 假设您已将 LlamaGen 目录安装为可编辑包 (`uv pip install -e .`)
from tokenizer.tokenizer_image.vq_model import VQ_models
from autoregressive.models.gpt import GPT_models
from autoregressive.models.generate import generate # 虽然此处用不到，但保留原始文件的导入
# ----------------------------------------

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with: uv pip install wandb")

# --- 辅助函数：加载模型和 VQ ---
def load_llama_gen_models(args, device):
    """Loads VQ and GPT models in LlamaGen style."""
    
    # 1. Load VQ Model (Image Tokenizer/Encoder/Decoder)
    vq_model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim)
    vq_model.to(device)
    vq_model.eval()
    checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
    vq_model.load_state_dict(checkpoint["model"])
    del checkpoint
    print(f"✓ Image tokenizer (VQ Model) is loaded")

    # 2. Load GPT Model (Autoregressive Generator)
    precision = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.precision]
    latent_size = args.image_size // args.downsample_size
    gpt_model = GPT_models[args.gpt_model](
        vocab_size=args.codebook_size,
        block_size=latent_size ** 2,
        num_classes=args.num_classes,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
    ).to(device=device, dtype=precision)
    
    checkpoint = torch.load(args.gpt_ckpt, map_location="cpu")
    # Using LlamaGen style checkpoint loading logic
    if args.from_fsdp: # fspd
        model_weight = checkpoint
    elif "model" in checkpoint:  # ddp
        model_weight = checkpoint["model"]
    elif "module" in checkpoint: # deepspeed
        model_weight = checkpoint["module"]
    elif "state_dict" in checkpoint:
        model_weight = checkpoint["state_dict"]
    else:
        raise Exception("please check model weight, maybe add --from-fsdp to run command")
        
    gpt_model.load_state_dict(model_weight, strict=False)
    gpt_model.eval()
    del checkpoint
    print(f"✓ GPT model is loaded")
    
    return vq_model, gpt_model, latent_size


class ProtoTokenOptimizer:
    """Optimizer class for learning proto-tokens"""
    
    def __init__(self, model, hidden_size, device, num_steps=1000, use_wandb=False):
        self.model = model
        self.device = device
        self.hidden_size = hidden_size
        self.use_wandb = use_wandb and WANDB_AVAILABLE

        # Freeze model parameters to ensure they are not updated and save memory
        self.model.requires_grad_(False)

        # infer dtype from model parameters (safer than model.dtype)
        model_param = next(self.model.parameters())
        self.dtype = model_param.dtype

        # Initialize proto-tokens (learnable embeddings)
        # e_t: image specific embedding
        self.e_t = nn.Parameter(torch.randn(1, hidden_size, device=device, dtype=self.dtype) * 0.01)
        # m: reusable embedding
        self.m = nn.Parameter(torch.randn(1, hidden_size, device=device, dtype=self.dtype) * 0.01)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            [self.e_t, self.m],
            lr=0.01, # args.learning_rate is used in main, but hardcoded here as per original code
            weight_decay=0.0
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=num_steps,
            eta_min=1e-5
        )
        
        # Record training history
        self.history = {
            'loss': [], 'accuracy': [], 'correct_tokens': [], 'top5_accuracy': [],
            'e_t_norm': [], 'm_norm': [], 'grad_norm': []
        }
        
    # LlamaGen/GPT_model 输入的第一个维度是批量大小 (Batch Size)，这里为 1
    def construct_input_embeddings(self, seq_length):
        """
        Construct input sequence: [e_t, m, m, m, ..., m]
        
        Args:
            seq_length: Target sequence length
        
        Returns:
            input_embeds: [1, seq_length, hidden_size]
        """
        # 第一位是 e_t，其余是 m
        if seq_length < 1:
            raise ValueError("seq_length must be >= 1")
        if seq_length == 1:
            input_embeds = self.e_t
        else:
            m_repeated = self.m.repeat(seq_length - 1, 1)  # [seq_length-1, hidden_size]
            input_embeds = torch.cat([self.e_t, m_repeated], dim=0)  # [seq_length, hidden_size]

        # LlamaGen/GPT_model 的输入是 (B, L, H)
        return input_embeds.unsqueeze(0).to(device=self.device, dtype=self.dtype)  # [1, seq_length, hidden_size]

    def _forward_embeddings_through_model(self, input_embeds, input_pos=None):
        """Forward the provided embeddings through the Transformer's layers and output logits.

        This replicates the internal forward pass but accepts raw embeddings (B, L, H).
        Returns logits of shape (B, L, vocab_size).
        """
        model = self.model
        # move to model device/dtype
        device = next(model.parameters()).device
        input_embeds = input_embeds.to(device=device, dtype=next(model.parameters()).dtype)

        # apply token dropout / embedding dropout
        h = model.tok_dropout(input_embeds)

        # prepare input_pos (expected by layers). Use a tensor of positions when not provided
        if input_pos is None:
            input_pos = torch.arange(0, h.shape[1], device=device)

        # determine freqs_cis to use in the layers
        # ensure freqs_cis tensor is on the same device as inputs to avoid CPU/GPU indexing errors
        if hasattr(model, 'freqs_cis') and model.freqs_cis is not None:
            freqs_all = model.freqs_cis.to(device)
        else:
            freqs_all = None

        if model.training:
            freqs_cis = freqs_all[: h.shape[1]] if freqs_all is not None else None
        else:
            # model.freqs_cis indexed by positions
            freqs_cis = freqs_all[input_pos] if freqs_all is not None else None

        # pass through transformer blocks
        # KV cache tensors may be marked as "inference tensors" (created under
        # inference_mode). Updating them inplace will fail unless we operate
        # inside inference_mode; we need gradients for proto-tokens, so instead
        # convert those buffers to normal tensors by cloning them here.
        for b in model.layers:
            if hasattr(b.attention, 'kv_cache') and b.attention.kv_cache is not None:
                try:
                    b.attention.kv_cache.k_cache = b.attention.kv_cache.k_cache.clone()
                    b.attention.kv_cache.v_cache = b.attention.kv_cache.v_cache.clone()
                except Exception:
                    # If cloning fails for any reason, continue; the model may still work.
                    pass

        for layer in model.layers:
            h = layer(h, freqs_cis, input_pos, mask=None)

        h = model.norm(h)
        logits = model.output(h).float()
        return logits
    
    def compute_loss_and_accuracy(self, target_tokens):
        """
        Compute loss and accuracy
        
        Args:
            target_tokens: [1, seq_length] Target token sequence (Visual Tokens)
        
        Returns:
            loss, accuracy, correct_count, top5_acc
        """
        seq_length = target_tokens.size(1)

        # Construct input embeddings
        input_embeds = self.construct_input_embeddings(seq_length)

        # Forward embeddings through the transformer's layers to get logits
        logits = self._forward_embeddings_through_model(input_embeds)

        # logits: [B, L, vocab_size]
        vocab_size = logits.size(-1)

        # move target tokens to model device
        target = target_tokens.to(logits.device)

        # Compute loss
        loss = F.cross_entropy(
            logits.view(-1, vocab_size),
            target.view(-1),
            reduction='mean'
        )

        # Compute accuracy
        predictions = logits.argmax(dim=-1)
        correct = (predictions == target).sum().item()
        total = target.numel()
        accuracy = correct / total * 100

        # Compute Top-5 accuracy
        top5_preds = logits.topk(5, dim=-1).indices
        top5_correct = (top5_preds == target.unsqueeze(-1)).any(dim=-1).sum().item()
        top5_accuracy = top5_correct / total * 100

        return loss, accuracy, correct, top5_accuracy

    # ... (train_step, evaluate, generate 方法与 SimpleAR 代码保持一致，因为它们是优化器逻辑)
    # ---------------------------------------------------------------------------------------
    def train_step(self, target_tokens):
        """Execute one training step"""
        self.model.eval()  # Keep model frozen
        self.optimizer.zero_grad()
        
        loss, accuracy, correct, top5_acc = self.compute_loss_and_accuracy(target_tokens)

        loss.backward()
        
        # Compute gradient norm (before clipping)
        grad_norm = torch.sqrt(
            sum(p.grad.norm() ** 2 for p in [self.e_t, self.m] if p.grad is not None)
        ).item()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_([self.e_t, self.m], max_norm=1.0)
        
        self.optimizer.step()
        self.scheduler.step()
        
        # Compute embedding norms
        e_t_norm = self.e_t.norm().item()
        m_norm = self.m.norm().item()
        
        # Record history
        self.history['loss'].append(loss.item())
        self.history['accuracy'].append(accuracy)
        self.history['correct_tokens'].append(correct)
        self.history['top5_accuracy'].append(top5_acc)
        self.history['e_t_norm'].append(e_t_norm)
        self.history['m_norm'].append(m_norm)
        self.history['grad_norm'].append(grad_norm)
        
        return loss.item(), accuracy, correct, top5_acc, e_t_norm, m_norm, grad_norm

    @torch.no_grad()
    def evaluate(self, target_tokens):
        """Evaluate reconstruction effect of current proto-tokens"""
        self.model.eval()
        return self.compute_loss_and_accuracy(target_tokens)
    
    @torch.no_grad()
    def generate(self, seq_length):
        """Generate sequence using current proto-tokens"""
        input_embeds = self.construct_input_embeddings(seq_length)
        logits = self._forward_embeddings_through_model(input_embeds)
        # Greedy decoding
        predictions = logits.argmax(dim=-1)
        return predictions.to(torch.long)
    # ---------------------------------------------------------------------------------------


def get_reference_tokens(vq_model, gpt_model, args):
    """
    Generate a blank image/tokens or load an existing image's tokens
    
    In LlamaGen C2I setting, we need an image to get the target visual tokens.
    Since LlamaGen sample.py is Class-to-Image, we use a synthetic image 
    or ask the user to provide an image path for the target tokens.
    
    Here, we will generate a blank image (all tokens = 0) as a placeholder target for feasibility test.
    For a real test, the user must provide an image path to encode.
    """
    print("\n" + "="*80)
    print("Step 1: Generate/Encode Target Visual Tokens (Placeholder)")
    print("="*80)
    
    # Class-conditional generation: use GPT to sample visual tokens conditioned on a class id
    latent_size = args.image_size // args.downsample_size
    seq_length = latent_size ** 2

    print(f"Target sequence length: {seq_length}")

    # prepare conditioning class ids (use same default list as sample_c2i.py)
    device = next(gpt_model.parameters()).device
    if getattr(args, 'class_id', None) is not None:
        class_labels = [int(args.class_id)]
    else:
        # default to a single class (first of sample list) to generate one image
        class_labels = [207]

    print(f"Generating image for class id: {class_labels[0]}")

    c_indices = torch.tensor(class_labels, device=device)

    # sample visual token indices with the repo's generate() function (match sample_c2i behavior)
    with torch.inference_mode():
        index_sample = generate(
            gpt_model, c_indices, seq_length,
            cfg_scale=getattr(args, 'cfg_scale', 1.0),
            cfg_interval=getattr(args, 'cfg_interval', -1),
            temperature=getattr(args, 'temperature', 1.0),
            top_k=getattr(args, 'top_k', 0),
            top_p=getattr(args, 'top_p', 1.0),
            sample_logits=True,
        )

    # index_sample: [batch=len(class_labels), seq_length]
    # decode to images via VQ model
    qzshape = [index_sample.shape[0], args.codebook_embed_dim, latent_size, latent_size]
    vq_dev = next(vq_model.parameters()).device
    index_sample = index_sample.to(vq_dev)

    with torch.inference_mode():
        index_for_decode = index_sample.reshape(-1, latent_size, latent_size).unsqueeze(1)
        # call decode_code with shape parameter matching repo usage
        reference_image = vq_model.decode_code(index_for_decode, shape=qzshape)

    print(f"✓ Generated visual tokens shape: {index_sample.shape}")
    return index_sample, reference_image


def verify_proto_tokens(model, visual_tokens, args):
    """
    Verify if proto-tokens can reconstruct visual token sequence
    (Same as original SimpleAR implementation, but without tokenizer_offset)
    """
    print("\n" + "="*80)
    print("Step 2: Optimize Proto-Tokens to reconstruct visual sequence")
    print("="*80)
    
    seq_length = visual_tokens.size(1)
    # Determine hidden size from model config (repo uses `config.dim`)
    if hasattr(model, 'config'):
        hidden_size = getattr(model.config, 'hidden_size', None) or getattr(model.config, 'dim', None)
    else:
        hidden_size = getattr(model, 'embed_dim', None) or getattr(model, 'd_model', None)
    if hidden_size is None:
        raise RuntimeError('Could not determine model hidden size (expected model.config.dim or model.embed_dim)')
    
    print(f"Sequence length: {seq_length}")
    print(f"Hidden size: {hidden_size}")
    print(f"Optimization steps: {args.num_steps}")
    
    # Create optimizer
    optimizer = ProtoTokenOptimizer(
        model=model,
        hidden_size=hidden_size,
        device=args.device,
        num_steps=args.num_steps,
        use_wandb=args.use_wandb
    )
    
    best_accuracy = 0
    best_step = 0
    pbar = tqdm(range(args.num_steps), desc="Optimizing Proto-Tokens")
    for step in pbar:
        loss, accuracy, correct, top5_acc, e_t_norm, m_norm, grad_norm = optimizer.train_step(visual_tokens)
        
        pbar.set_postfix({'loss': f'{loss:.4f}', 'acc': f'{accuracy:.2f}%', 'correct': f'{correct}/{seq_length}', 'top5': f'{top5_acc:.2f}%'})
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_step = step
        
        if args.use_wandb and WANDB_AVAILABLE:
            log_dict = {
                'train/loss': loss, 'train/accuracy': accuracy, 'train/top5_accuracy': top5_acc,
                'train/correct_tokens': correct, 'train/total_tokens': seq_length,
                'train/best_accuracy': best_accuracy,
                'train/learning_rate': optimizer.scheduler.get_last_lr()[0],
                'embeddings/e_t_norm': e_t_norm, 'embeddings/m_norm': m_norm,
                'gradients/grad_norm': grad_norm, 'step': step
            }
            wandb.log(log_dict)
        
        if (step + 1) % 100 == 0:
            print(f"\n[Step {step+1}/{args.num_steps}]")
            print(f"  Accuracy: {accuracy:.2f}% ({correct}/{seq_length} tokens)")
            print(f"  Best Accuracy: {best_accuracy:.2f}% (at step {best_step})")
            
    # Save proto-tokens
    proto_tokens_path = os.path.join(args.save_dir, "proto_tokens.pt")
    torch.save({
        'e_t': optimizer.e_t.detach().cpu(),
        'm': optimizer.m.detach().cpu(),
        'target_tokens': visual_tokens.cpu(),
        'history': optimizer.history,
        'best_accuracy': best_accuracy,
        'seq_length': seq_length
    }, proto_tokens_path)
    print(f"Proto-tokens saved to: {proto_tokens_path}")
    
    # WandB save artifact
    if args.use_wandb and WANDB_AVAILABLE:
        artifact = wandb.Artifact('proto_tokens', type='model')
        artifact.add_file(proto_tokens_path)
        wandb.log_artifact(artifact)
    
    return optimizer


def reconstruct_image(optimizer, vq_model, visual_tokens, latent_size, args):
    """
    Reconstruct image using learned proto-tokens
    """
    print("\n" + "="*80)
    print("Step 3: Reconstruct image using Proto-Tokens")
    print("="*80)
    
    seq_length = visual_tokens.size(1)
    
    # Generate reconstructed tokens
    reconstructed_tokens = optimizer.generate(seq_length)
    
    # Compute reconstruction accuracy
    correct = (reconstructed_tokens == visual_tokens).sum().item()
    total = seq_length
    accuracy = correct / total * 100
    
    print(f"Reconstruction Accuracy: {accuracy:.2f}% ({correct}/{total} tokens)")
    
    # Decode to image (LlamaGen VQ model decode)
    index_sample = reconstructed_tokens
    index_sample = index_sample.reshape(-1, latent_size, latent_size).unsqueeze(1)
    
    with torch.inference_mode():
        reconstructed_image = vq_model.decode_code(index_sample, shape=[1, args.codebook_embed_dim, latent_size, latent_size])
    
    print("✓ Image reconstruction completed")
    
    return reconstructed_image, reconstructed_tokens


def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = args.device

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Initialize WandB
    if args.use_wandb and WANDB_AVAILABLE:
        # ... (WandB initialization remains the same) ...
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                'gpt_model': args.gpt_model, 'vq_model': args.vq_model,
                'image_size': args.image_size, 'num_steps': args.num_steps,
                'learning_rate': args.learning_rate, 'seed': args.seed,
            },
            tags=['proto-tokens', 'feasibility', f'size-{args.image_size}']
        )
        print(f"✓ WandB initialized: {wandb.run.url}")
    
    print("="*80)
    print("Proto-Tokens Feasibility Verification Experiment (LlamaGen)")
    print("="*80)
    
    # Load VQ and GPT Models
    vq_model, gpt_model, latent_size = load_llama_gen_models(args, device)
    
    # Step 1: Encode a target image's visual tokens
    # Note: LlamaGen C2I sample code doesn't have an easy T2I path. 
    # We use a placeholder token sequence instead of generating/encoding a real image.
    tokens_path = os.path.join(args.save_dir, "visual_tokens_target.pt")
    
    if os.path.exists(tokens_path) and not args.force_generate:
        print(f"\nFound existing visual tokens at {tokens_path}. Skipping generation.")
        visual_tokens = torch.load(tokens_path, map_location=args.device)
        # Directly decode the saved visual tokens to get a reference image
        index_sample = visual_tokens.reshape(-1, latent_size, latent_size).unsqueeze(1)
        with torch.inference_mode():
            reference_image = vq_model.decode_code(index_sample, shape=[visual_tokens.size(0), args.codebook_embed_dim, latent_size, latent_size])
    else:
        # Generate class-conditional image tokens and reference image via GPT + VQ
        visual_tokens, reference_image = get_reference_tokens(vq_model, gpt_model, args)
        torch.save(visual_tokens, tokens_path)
        print(f"Visual tokens saved to: {tokens_path}")

    # Save reference image (visualization of the target tokens)
    ref_image_path = os.path.join(args.save_dir, "target_image.png")
    save_image(reference_image, ref_image_path, normalize=True, value_range=(-1, 1))
    print(f"Target/Reference image saved to: {ref_image_path}")
    
    # WandB record reference image
    if args.use_wandb and WANDB_AVAILABLE:
        # ... (WandB logging remains the same) ...
        ref_img_vis = (reference_image[0].float().cpu() + 1) / 2
        ref_img_vis = torch.clamp(ref_img_vis, 0, 1)
        
        wandb.log({
            "target_image": wandb.Image(
                ref_img_vis,
                caption="Target Image (Visualization of Target Tokens)"
            ),
            "num_visual_tokens": visual_tokens.size(1)
        })

    # Step 2: Optimize proto-tokens
    optimizer = verify_proto_tokens(gpt_model, visual_tokens, args)
    
    # Step 3: Reconstruct image
    reconstructed_image, reconstructed_tokens = reconstruct_image(
        optimizer, vq_model, visual_tokens, latent_size, args
    )
    
    # Save reconstructed image
    recon_image_path = os.path.join(args.save_dir, "reconstructed_image.png")
    save_image(reconstructed_image, recon_image_path, normalize=True, value_range=(-1, 1))
    print(f"Reconstructed image saved to: {recon_image_path}")
    
    # Create comparison image
    comparison = torch.cat([reference_image, reconstructed_image], dim=-1)
    comp_path = os.path.join(args.save_dir, "comparison.png")
    save_image(comparison, comp_path, normalize=True, value_range=(-1, 1))
    print(f"Comparison image saved to: {comp_path}")
    
    # WandB record reconstruction result and Final summary
    # ... (Final logging and summary logic remains the same) ...
    if args.use_wandb and WANDB_AVAILABLE:
        recon_img_vis = (reconstructed_image[0].float().cpu() + 1) / 2
        recon_img_vis = torch.clamp(recon_img_vis, 0, 1)
        comp_img_vis = (comparison[0].float().cpu() + 1) / 2
        comp_img_vis = torch.clamp(comp_img_vis, 0, 1)
        
        wandb.log({
            "reconstructed_image": wandb.Image(
                recon_img_vis,
                caption=f"Reconstructed (Acc: {optimizer.history['accuracy'][-1]:.2f}%)"
            ),
            "comparison": wandb.Image(comp_img_vis, caption="Left: Target | Right: Reconstructed"),
        })
    
    # Final summary (same as original)
    print("\n" + "="*80)
    print("Experiment Summary")
    print("="*80)
    print(f"✓ Sequence Length: {visual_tokens.size(1)} tokens")
    print(f"✓ Final Accuracy: {optimizer.history['accuracy'][-1]:.2f}%")
    print(f"✓ Best Accuracy: {max(optimizer.history['accuracy']):.2f}%")
    
    if args.use_wandb and WANDB_AVAILABLE:
        # ... (wandb.summary update and finish) ...
        pass # Placeholder for brevity

    print("\nAll results saved to:", args.save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify Proto-Tokens Feasibility for LlamaGen-like GPT")
    
    # --- LlamaGen Model related (based on sample.py) ---
    parser.add_argument("--gpt-model", type=str, choices=list(GPT_models.keys()), default="GPT-B")
    parser.add_argument("--gpt-ckpt", type=str, required=True, help="ckpt path for GPT model")
    parser.add_argument("--gpt-type", type=str, choices=['c2i', 't2i'], default="c2i", help="class-conditional or text-conditional")
    parser.add_argument("--from-fsdp", action='store_true')
    parser.add_argument("--cls-token-num", type=int, default=1, help="max token number of condition input")
    parser.add_argument("--precision", type=str, default='bf16', choices=["none", "fp16", "bf16"]) 
    
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str, required=True, help="ckpt path for vq model")
    parser.add_argument("--codebook-size", type=int, default=16384, help="codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="codebook dimension for vector quantization")
    parser.add_argument("--image-size", type=int, choices=[256, 384, 512], default=384)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0, help="classifier-free guidance scale")
    parser.add_argument("--cfg-interval", type=int, default=-1, help="classifier-free guidance interval")
    parser.add_argument("--top-k", type=int, default=2000, help="top-k value to sample with")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature value to sample with")
    parser.add_argument("--top-p", type=float, default=1.0, help="top-p value to sample with")
    parser.add_argument("--class-id", type=int, default=None, help="Class id to condition generation on (default: first from sample list)")
    
    # --- Optimization related (from SimpleAR code) ---
    parser.add_argument("--num-steps", type=int, default=1000, help="Optimization steps")
    parser.add_argument("--learning-rate", type=float, default=0.01, help="Learning rate (Note: currently hardcoded to 0.01 in optimizer init)")
    
    # --- Target Generation related (Modified) ---
    parser.add_argument("--force-generate", action="store_true", help="Force generate/re-encode target tokens")
    
    # --- Others ---
    parser.add_argument("--save-dir", type=str, default="./proto_tokens_exp_llama", help="Directory to save results")
    parser.add_argument("--device", type=str, default="cuda", help="Computing device (Default is cuda)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # --- WandB related ---
    parser.add_argument("--use-wandb", action="store_true", help="Use WandB to record experiment")
    parser.add_argument("--wandb-project", type=str, default="llmagen-proto-tokens", help="WandB project name")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="WandB run name")
    
    # Use parse_known_args to be tolerant if this script is invoked in environments
    # where additional sampling args might be provided elsewhere. Unknown args
    # will be parsed by a secondary parser that handles sampling options.
    args, unknown = parser.parse_known_args()
    if unknown:
        extra_parser = argparse.ArgumentParser(add_help=False)
        extra_parser.add_argument("--cfg-scale", type=float, default=getattr(args, 'cfg_scale', 4.0))
        extra_parser.add_argument("--cfg-interval", type=int, default=getattr(args, 'cfg_interval', -1))
        extra_parser.add_argument("--top-k", type=int, default=getattr(args, 'top_k', 2000))
        extra_parser.add_argument("--temperature", type=float, default=getattr(args, 'temperature', 1.0))
        extra_parser.add_argument("--top-p", type=float, default=getattr(args, 'top_p', 1.0))
        extra_args, _ = extra_parser.parse_known_args(unknown)
        # attach any extra parsed sampling args back onto args
        for k, v in vars(extra_args).items():
            setattr(args, k, v)
    args.device = "cuda" if torch.cuda.is_available() and args.device == "cuda" else args.device
    
    if args.use_wandb and args.wandb_run_name is None:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.wandb_run_name = f"proto_tokens_{args.image_size}px_{timestamp}"
    
    main(args)