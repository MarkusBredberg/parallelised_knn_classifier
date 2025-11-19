# Parallel K-Nearest Neighbors Classifier with MPI

A high-performance parallel implementation of K-Nearest Neighbors (KNN) classification using spatial decomposition and MPI for distributed computing. Designed for radio galaxy classification tasks and scalability analysis.

## 🎯 Project Overview

This project implements a parallel KNN classifier that distributes data across multiple MPI processes using spatial decomposition techniques. The implementation demonstrates efficient parallel programming patterns including:

- **Spatial domain decomposition** with ghost region exchange
- **Multi-node execution** with inter-process communication
- **Load balancing** analysis and optimization strategies
- **Performance profiling** with strong and weak scaling experiments

## ✨ Key Features

### Parallel Architecture
- **1D Spatial Decomposition**: Divides embedding space along first dimension
- **Ghost Region Exchange**: Uses MPI `sendrecv` for boundary data communication (similar to Poisson solver patterns)
- **Flexible Partitioning**: Supports both uniform spatial and data-driven balanced partitioning
- **Multi-Node Support**: Tracks communication overhead across cluster nodes

### Data Processing
- **Synthetic Data Generation**: Creates clustered embeddings with configurable class separation
- **Random Orthogonal Rotation**: Ensures uniform class distribution across processes
- **Pre-trained Encoder Integration**: Compatible with BYOL/DINO models for real radio galaxy data

### Performance Analysis
- **Strong Scaling**: Fixed problem size across varying process counts
- **Weak Scaling**: Problem size scales proportionally with processes
- **Comprehensive Metrics**: Speedup, efficiency, load balance, communication overhead
- **Visualization Suite**: Automated plotting of all performance metrics

## 📊 Performance Results

Typical results on 2-node cluster (synthetic data, 800 training / 200 test samples):

| Processes | Nodes | Speedup | Efficiency | Accuracy |
|-----------|-------|---------|------------|----------|
| 1         | 1     | 1.00x   | 100%       | ~99%     |
| 2         | 2     | 2.08x   | 104%       | ~99%     |
| 4         | 2     | 3.17x   | 79%        | ~97%     |
| 8         | 2     | 4.17x   | 52%        | ~95%     |

*Note: Communication overhead increases with process count, especially across nodes*

## 🚀 Quick Start

### Prerequisites
```bash
# Required packages
pip install numpy mpi4py matplotlib scikit-learn scipy
```

### Running the Complete Pipeline
```bash
# Generate data, run experiments, create plots
python run_full_pipeline.py --procs 1 2 4 8

# Or with MPI on a cluster (via Slurm)
sbatch run_knn_cluster.job
```

### Individual Steps
```bash
# Step 1: Generate synthetic training/test data
python step1_generate_features.py

# Step 2: Run parallel KNN classification
mpirun -np 4 python step2_parallel_knn.py

# Step 3: Visualize performance results
python step2.1_plot_results.py
```

## 📁 Project Structure
```
parallelised_knn_classifier/
├── step1_generate_features.py      # Data generation with class clustering
├── step1.1_plot_data.py            # Visualize data distribution
├── step2_parallel_knn.py           # Main parallel KNN implementation
├── step2.1_plot_results.py         # Performance analysis plots
├── run_full_pipeline.py            # Master pipeline script
├── profiling_script.py             # Serial bottleneck analysis
├── run_knn_cluster.job             # Slurm batch script for cluster
├── produced_data/
│   ├── strong_scaling/             # Fixed problem size results
│   └── weak_scaling/               # Scaling problem size results
└── figures/                        # Generated performance plots
```

## 🔧 Configuration Options

### Data Generation
```bash
# Adjust class separation (default: 1.0)
python step1_generate_features.py --cluster-strength 2.0

# Generate for specific process counts
python step1_generate_features.py --procs 1 2 4 8 16

# Use balanced partitioning (equal points per process)
python step1_generate_features.py --balanced
```

### Parallel Execution
```bash
# Strong scaling (fixed problem size)
mpirun -np 8 python step2_parallel_knn.py

# Weak scaling (problem size scales with processes)
mpirun -np 8 python step2_parallel_knn.py --weak-scaling

# Balanced partitioning mode
mpirun -np 8 python step2_parallel_knn.py --balanced
```

## 🏗️ Implementation Details

### Spatial Decomposition Strategy

The classifier divides the N-dimensional embedding space along the first dimension:
```
Process 0: [x_min, x_min + Δx)
Process 1: [x_min + Δx, x_min + 2Δx)
...
Process P-1: [x_min + (P-1)Δx, x_max]
```

Each process:
1. Owns a spatial domain of the embedding space
2. Stores local training points
3. Exchanges ghost regions (15% overlap) with neighbors
4. Classifies queries in its domain using local + ghost data

### Communication Pattern
```
Rank 0 ←→ Rank 1 ←→ Rank 2 ←→ ... ←→ Rank P-1

Each process exchanges boundary points with 2 neighbors (left/right)
Ghost width: 15% of domain size
MPI primitive: sendrecv (blocking, avoids deadlock)
```

### Load Balance Considerations

**Challenge**: If classes cluster spatially, uniform partitioning creates imbalance

**Solution**: Random orthogonal rotation of embedding space
- Preserves distance relationships (KNN accuracy maintained)
- Mixes classes across spatial domains
- Ensures all processes can classify all classes

## 📈 Performance Metrics

The analysis suite computes:

1. **Speedup**: T₁ / Tₚ (ideal: linear with P)
2. **Efficiency**: Speedup / P (ideal: 100%)
3. **Load Balance**: max(points) / min(points) across processes
4. **Communication Overhead**: Ghost exchange time per process
5. **Classification Accuracy**: Correctness vs serial baseline

## 🎓 Academic Context

This project was developed for **PHYS 743: Parallel Programming** to demonstrate:
- Domain decomposition techniques from numerical methods (Poisson solver)
- Extended to machine learning applications (KNN classification)
- Performance analysis on multi-node HPC clusters
- Real-world application to radio galaxy classification

## 🛠️ Cluster Setup (Helvetios)

For running on EPFL's Helvetios cluster:
```bash
# Load required modules
module load intel/2021.6.0
module load intel-oneapi-mpi/2021.6.0
module load python/3.10.4

# Create virtual environment
python3 -m venv venv_knn
source venv_knn/bin/activate
pip install mpi4py numpy matplotlib scikit-learn scipy

# Submit job
sbatch run_knn_cluster.job
```

## 📊 Generated Outputs

### Data Visualizations
- `data_overview.png` - Training/test distribution with PCA/t-SNE
- `domain_decomposition.png` - Spatial partitioning across processes
- `class_distribution.png` - Class balance per process

### Performance Plots
- `strong_speedup_analysis.png` - Speedup vs process count
- `strong_efficiency_analysis.png` - Parallel efficiency
- `strong_load_balance_analysis.png` - Training point distribution
- `weak_scaling_analysis.png` - Execution time with scaling problem
- `communication_overhead.png` - Ghost exchange timing

## 🔍 Debugging Features

The implementation includes extensive instrumentation:
```python
# Hostname tracking
hostname = socket.gethostname()
all_hostnames = comm.gather(hostname, root=0)

# Communication timing
comm_start = MPI.Wtime()
ghost_data = comm.sendrecv(...)
comm_time = MPI.Wtime() - comm_start

# Load balance statistics
all_training_counts = comm.gather(len(local_training_data), root=0)
```

## 🐛 Known Limitations

1. **Accuracy Degradation**: May occur at high process counts if classes cluster spatially
2. **1D Decomposition**: Only divides along first embedding dimension
3. **Static Load Balance**: Determined at initialization, no dynamic rebalancing
4. **Ghost Width**: Fixed at 15% of domain, may need tuning for different datasets

## 📚 References

- MPI-based domain decomposition: Similar patterns to course Poisson solver
- KNN algorithm: Scikit-learn documentation
- Radio galaxy classification: [Relevant astronomy papers if applicable]

## 👤 Author

**Markus Bredberg**  
EPFL - PHYS 743: Parallel Programming

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- PHYS 743 course materials and instructors
- EPFL Helvetios cluster for computational resources
- Pre-trained SSL models from radio astronomy research community

---

**Note**: This is an educational project demonstrating parallel programming concepts. For production use, consider more sophisticated partitioning strategies and load balancing techniques.
