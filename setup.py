from setuptools import find_packages, setup
setup(name='fame-mapping', version='0.1.0', packages=find_packages(), python_requires='>=3.10', install_requires=['torch', 'torchvision', 'pytorch3d', 'numpy', 'scipy', 'trimesh', 'matplotlib', 'Pillow', 'plotly', 'scikit-learn', 'tqdm', 'rtree'], entry_points={'console_scripts': ['fame=fame.cli:main']})
