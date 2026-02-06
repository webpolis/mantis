"""
HMST: Hierarchical Memory-State Transformer

Setup script for installation.
"""

from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='hmst',
    version='1.0.0',
    author='Claude (Anthropic) & Gemini (Google)',
    description='Hierarchical Memory-State Transformer - A novel LLM architecture for mitigating hallucination',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
        'torch>=2.10.0',
        'numpy>=2.4.2',
        'faiss-cpu>=1.13.2',
        'transformers>=5.0.0',
        'tokenizers>=0.22.2',
        'tqdm>=4.67.3',
    ],
    extras_require={
        'dev': [
            'pytest>=9.0.2',
            'black>=26.1.0',
            'flake8>=7.3.0',
        ],
        'wandb': ['wandb>=0.24.2'],
        'distributed': [
            'deepspeed>=0.18.5',
            'accelerate>=1.12.0',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
