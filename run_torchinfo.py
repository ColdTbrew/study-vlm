from graphviz import Digraph

dot = Digraph(comment="LLaVA 1.5 7B – Submodule Architecture")

# 전체 방향 (위→아래 or 왼→오 원하는 쪽으로)
dot.attr(rankdir="TB", fontsize="10")
dot.attr("node", shape="record", style="rounded,filled", fillcolor="#eef2ff")

# 1) 최상위
dot.node("LlavaForConditionalGeneration",
         "{LlavaForConditionalGeneration|lm_head: Linear(4096 → 32064)}")

# 2) LlavaModel
dot.node("LlavaModel", "LlavaModel")
dot.edge("LlavaForConditionalGeneration", "LlavaModel")

# 3) Vision Tower 쪽
dot.node("VisionTower", "{vision_tower|CLIPVisionModel}")
dot.edge("LlavaModel", "VisionTower")

dot.node("CLIPVisionTransformer",
         "{CLIPVisionTransformer|embeddings|pre_layernorm|encoder (24×)|post_layernorm}")
dot.edge("VisionTower", "CLIPVisionTransformer")

dot.node("CLIPEmb",
         "{CLIPVisionEmbeddings|patch_embedding: Conv2d(3→1024, 14×14, s=14)|position_embedding: Embedding(577, 1024)}")
dot.edge("CLIPVisionTransformer", "CLIPEmb")

dot.node("CLIPEncoder", "{CLIPEncoder|layers: 24 × CLIPEncoderLayer}")
dot.edge("CLIPVisionTransformer", "CLIPEncoder")

dot.node("CLIPEncoderLayer",
         "{CLIPEncoderLayer|self_attn: CLIPAttention|mlp: CLIPMLP|LayerNorm ×2}")
dot.edge("CLIPEncoder", "CLIPEncoderLayer")

dot.node("CLIPAttention",
         "{CLIPAttention|q_proj: Linear(1024→1024)|k_proj: Linear(1024→1024)|v_proj: Linear(1024→1024)|out_proj: Linear(1024→1024)}")
dot.edge("CLIPEncoderLayer", "CLIPAttention")

dot.node("CLIPMLP",
         "{CLIPMLP|fc1: Linear(1024→4096)|fc2: Linear(4096→1024)|QuickGELU}")
dot.edge("CLIPEncoderLayer", "CLIPMLP")

# 4) Multi-Modal Projector
dot.node("Projector",
         "{multi_modal_projector|Linear(1024→4096)|GELU|Linear(4096→4096)}")
dot.edge("LlavaModel", "Projector")

# 5) Language Model (LLaMA)
dot.node("LlamaModel",
         "{language_model: LlamaModel|embed_tokens: Embedding(32064, 4096)|norm: RMSNorm(4096)|rotary_emb: RoPE}")
dot.edge("LlavaModel", "LlamaModel")

dot.node("LlamaLayers", "layers: 32 × LlamaDecoderLayer")
dot.edge("LlamaModel", "LlamaLayers")

dot.node("LlamaDecoderLayer",
         "{LlamaDecoderLayer|self_attn: LlamaAttention|mlp: LlamaMLP|input_norm|post_attn_norm}")
dot.edge("LlamaLayers", "LlamaDecoderLayer")

dot.node("LlamaAttention",
         "{LlamaAttention|q_proj: Linear(4096→4096)|k_proj: Linear(4096→4096)|v_proj: Linear(4096→4096)|o_proj: Linear(4096→4096)}")
dot.edge("LlamaDecoderLayer", "LlamaAttention")

dot.node("LlamaMLP",
         "{LlamaMLP|gate_proj: Linear(4096→11008)|up_proj: Linear(4096→11008)|down_proj: Linear(11008→4096)|SiLU}")
dot.edge("LlamaDecoderLayer", "LlamaMLP")

# 6) 파일로 저장
dot.render("llava_submodules", format="png", cleanup=True)
print("saved: llava_submodules.png")