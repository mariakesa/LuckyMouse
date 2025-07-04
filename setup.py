from setuptools import setup, find_packages

setup(
    name='pixel-transformer-neuro',
    version='0.1.0',
    description='A project for modeling neural responses with a Pixel Transformer',
    author='Maria Kesa',
    author_email='maria.kesa@example.com',
    packages=find_packages(include=['pixel_transformer_neuro', 'pixel_transformer_neuro.*']),
    include_package_data=True,
    install_requires=[
        'torch',
        'numpy',
        'scikit-learn',
        'matplotlib',
        'pandas',
        'wandb',
        'tqdm',
        # Add others as needed
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
