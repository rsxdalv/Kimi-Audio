from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import pathlib

# Get CUDA extension source files
cuda_dir = pathlib.Path("kimia_infer/models/detokenizer/vocoder/alias_free_activation/cuda")
sources = [
    str(cuda_dir / "anti_alias_activation.cpp"),
    str(cuda_dir / "anti_alias_activation_cuda.cu"),
]

# CUDA compilation flags
extra_compile_args = {
    "cxx": ["-O3"],
    "nvcc": [
        "-O3",
        "-gencode", "arch=compute_70,code=sm_70",
        "-gencode", "arch=compute_80,code=sm_80",
        "--use_fast_math",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
    ]
}

setup(
    name="kimia_infer",
    version="0.1.7",
    description="Kimi-Audio inference and toolkit package.",
    author="Moonshot AI",
    packages=find_packages(),
    install_requires=[
        "torch>=2.4.1",
        "torchaudio>=2.4.1",
        "packaging",
        "jinja2",
        "openai-whisper",
        "jsonlines",
        "pandas",
        "validators",
        "sty",
        "transformers",
        "librosa",
        "accelerate",
        "aiohttp",
        "colorama",
        "omegaconf>=2.3.0",
        "sox",
        "six>=1.16.0",
        "hyperpyyaml",
        "conformer>=0.3.2",
        "diffusers",
        "pillow",
        "sentencepiece",
        "easydict",
        "fire",
        "ujson",
        "cairosvg",
        "immutabledict",
        "rich",
        "wget",
        "gdown",
        "datasets",
        "torchdyn>=1.0.6",
        "huggingface_hub",
        "loguru",
        "decord",
        "blobfile",
        "timm",
        "sacrebleu>=1.5.1",
        "soundfile",
        "tqdm"
    ],
    include_package_data=True,
    package_data={
        'kimia_infer.models.detokenizer.vocoder.alias_free_activation': [
            'cuda/*.h',
            'cuda/*.cu',
            'cuda/*.cpp'
        ],
        'kimia_infer.models.tokenizer.whisper_Lv3': ['*.npz']
    },
    ext_modules=[
        CUDAExtension(
            name='kimia_infer.models.detokenizer.vocoder.alias_free_activation.cuda.anti_alias_activation_cuda',
            sources=sources,
            extra_compile_args=extra_compile_args,
            include_dirs=[str(cuda_dir)],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    python_requires=">=3.8",
)
