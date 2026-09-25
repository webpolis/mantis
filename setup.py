"""
MANTIS: Metacognitive Adaptive Network with Tiered Inference Strategies

Setup script for installation.
"""

from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='mantis',
    version='1.0.0',
    author='Nicolás Iglesias',
    author_email='nfiglesias@gmail.com',
    description='Metacognitive Adaptive Network with Tiered Inference Strategies - A novel LLM architecture for mitigating hallucination',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
        'torch>=2.10.0',
        'numpy>=2.4.2',
        'tqdm>=4.67.3',
        'accelerate>=1.12.0',
        'datasets>=4.5.0',
    ],
    extras_require={
        # Stage 2/3 memory systems and the full inference engine (mamba-ssm needs CUDA)
        'memory': [
            'mamba-ssm>=2.2.2',
            'faiss-cpu>=1.13.2',
        ],
        'dev': [
            'pytest>=9.0.2',
            'black>=26.1.0',
            'flake8>=7.3.0',
        ],
        'wandb': ['wandb>=0.24.2'],
        'distributed': [
            'deepspeed>=0.18.5',
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
