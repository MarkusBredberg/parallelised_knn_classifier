#!/usr/bin/env python3
"""
Master Pipeline Script for Parallel KNN Experiments
Runs the complete workflow: data generation → visualization → experiments → analysis

Usage:
    # Default: synthetic data, uniform partitioning, 1,2,4,8 processes
    python run_full_pipeline.py
    
    # Custom process counts
    python run_full_pipeline.py --procs 1 2 4 8 16
    
    # Increase class clustering (may improve accuracy)
    python run_full_pipeline.py --cluster-strength 2.0
    
    # Balanced partitioning
    python run_full_pipeline.py --balanced
    
    # Real data (if available)
    python run_full_pipeline.py --real
    
    # Skip data visualization (faster)
    python run_full_pipeline.py --skip-viz
    
    # Only run strong scaling (no weak scaling)
    python run_full_pipeline.py --strong-only
    
    # Only run weak scaling (no strong scaling)
    python run_full_pipeline.py --weak-only
"""

import subprocess
import sys
import argparse
from pathlib import Path
import time

class Colors:
    """ANSI color codes for pretty terminal output"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_section(title):
    """Print a nice section header"""
    print("\n" + "="*70)
    print(f"{Colors.HEADER}{Colors.BOLD}{title}{Colors.ENDC}")
    print("="*70 + "\n")


def print_step(step_num, total_steps, description):
    """Print step progress"""
    print(f"\n{Colors.OKBLUE}[Step {step_num}/{total_steps}] {description}{Colors.ENDC}")
    print("-"*70)


def run_command(cmd, description, check=True):
    """
    Run a shell command and handle output
    
    Parameters:
    -----------
    cmd : list
        Command and arguments
    description : str
        What this command does
    check : bool
        If True, raise error on non-zero exit
    """
    print(f"{Colors.OKCYAN}Running: {' '.join(cmd)}{Colors.ENDC}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, check=check, capture_output=False, text=True)
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"{Colors.OKGREEN}✓ {description} completed in {elapsed:.1f}s{Colors.ENDC}")
            return True
        else:
            print(f"{Colors.FAIL}✗ {description} failed with exit code {result.returncode}{Colors.ENDC}")
            return False
            
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"{Colors.FAIL}✗ {description} failed after {elapsed:.1f}s{Colors.ENDC}")
        print(f"{Colors.FAIL}Error: {e}{Colors.ENDC}")
        if check:
            raise
        return False
    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}⚠ Pipeline interrupted by user{Colors.ENDC}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Run complete parallel KNN pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Data source
    parser.add_argument('--real', action='store_true',
                       help='Use real radio galaxy images (default: synthetic)')
    parser.add_argument('--checkpoint', type=str,
                       default='/home/markusbredberg/Scripts/pretrained_model_weights/byol.ckpt',
                       help='Path to pre-trained encoder checkpoint (for --real)')
    parser.add_argument('--data-dir', type=str,
                       default='/home/markusbredberg/Scripts/data/firstgalaxydata/galaxy_data/all/',
                       help='Path to radio galaxy images (for --real)')
    
    # Synthetic data parameters
    parser.add_argument('--cluster-strength', type=float, default=1.0,
                       help='Controls class separation (default: 1.0, higher = more separated)')
    
    # Parallelization options
    parser.add_argument('--balanced', action='store_true',
                       help='Use balanced (data-driven) partitioning (default: uniform spatial)')
    parser.add_argument('--procs', type=int, nargs='+', default=[1, 2, 4, 8],
                       help='Process counts to test (default: 1 2 4 8)')
    
    # Scaling modes
    parser.add_argument('--strong-only', action='store_true',
                       help='Only run strong scaling experiments')
    parser.add_argument('--weak-only', action='store_true',
                       help='Only run weak scaling experiments')
    
    # Pipeline control
    parser.add_argument('--skip-viz', action='store_true',
                       help='Skip data visualization (faster)')
    parser.add_argument('--skip-data-gen', action='store_true',
                       help='Skip data generation (use existing data)')
    parser.add_argument('--skip-experiments', action='store_true',
                       help='Skip MPI experiments (only generate data and visualize)')
    
    # Output
    parser.add_argument('--out-dir', type=str, 
                       default='/home/markusbredberg/Scripts/parallelised_knn_classifier/produced_data',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Determine scaling modes
    if args.strong_only:
        run_strong = True
        run_weak = False
    elif args.weak_only:
        run_strong = False
        run_weak = True
    else:
        run_strong = True
        run_weak = True
    
    # Calculate total steps
    total_steps = 0
    total_steps += 0 if args.skip_data_gen else 1  # Data generation
    total_steps += 0 if args.skip_viz else 1       # Visualization
    if not args.skip_experiments:
        if run_strong:
            total_steps += len(args.procs)  # Strong scaling experiments
        if run_weak:
            total_steps += len(args.procs)  # Weak scaling experiments
        total_steps += 1  # Plot results
    
    current_step = 0
    
    # Print banner
    print_section("PARALLEL KNN PIPELINE")
    print(f"Data source: {'REAL radio galaxy images' if args.real else 'SYNTHETIC'}")
    if not args.real:
        print(f"Cluster strength: {args.cluster_strength}x (higher = more separated classes)")
    print(f"Partitioning: {'BALANCED (data-driven)' if args.balanced else 'UNIFORM SPATIAL'}")
    print(f"Process counts: {args.procs}")
    print(f"Scaling modes: {'Strong' if run_strong else ''}{' + ' if run_strong and run_weak else ''}{'Weak' if run_weak else ''}")
    print(f"Total steps: {total_steps}")
    
    pipeline_start = time.time()
    
    # ==================================================================
    # STEP 1: DATA GENERATION
    # ==================================================================
    if not args.skip_data_gen:
        current_step += 1
        print_step(current_step, total_steps, "Generating Data")
        
        cmd = ['python', 'step1_generate_features.py']
        
        if args.real:
            cmd.extend(['--real'])
            cmd.extend(['--checkpoint', args.checkpoint])
            cmd.extend(['--data-dir', args.data_dir])
        else:
            cmd.extend(['--cluster-strength', str(args.cluster_strength)])

        if run_weak == False:
            cmd.append('--no-weak-scaling')
        
        if args.balanced:
            cmd.append('--balanced')
        
        cmd.extend(['--out-dir', args.out_dir])
        cmd.extend(['--procs'] + [str(p) for p in args.procs])
        
        success = run_command(cmd, "Data generation")
        if not success:
            print(f"\n{Colors.FAIL}✗ Pipeline failed at data generation{Colors.ENDC}")
            return 1
    else:
        print(f"\n{Colors.WARNING}⊘ Skipping data generation (using existing data){Colors.ENDC}")
    
    # ==================================================================
    # STEP 2: DATA VISUALIZATION
    # ==================================================================
    if not args.skip_viz:
        current_step += 1
        print_step(current_step, total_steps, "Visualizing Data")
        
        cmd = ['python', 'step1.1_plot_data.py']
        
        success = run_command(cmd, "Data visualization", check=False)
        if not success:
            print(f"{Colors.WARNING}⚠ Visualization failed, but continuing...{Colors.ENDC}")
    else:
        print(f"\n{Colors.WARNING}⊘ Skipping data visualization{Colors.ENDC}")
    
    if args.skip_experiments:
        print_section("PIPELINE COMPLETE (Experiments Skipped)")
        elapsed = time.time() - pipeline_start
        print(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
        return 0
    
    # ==================================================================
    # STEP 3: STRONG SCALING EXPERIMENTS
    # ==================================================================
    if run_strong:
        print_section("STRONG SCALING EXPERIMENTS")
        
        for proc_count in args.procs:
            current_step += 1
            print_step(current_step, total_steps, 
                      f"Strong Scaling with {proc_count} processes")
            
            cmd = ['mpirun', '-np', str(proc_count), 
                   'python', 'step2_parallel_knn.py',
                   '--out-dir', args.out_dir]
            
            if args.balanced:
                cmd.append('--balanced')
            
            success = run_command(cmd, 
                                f"Strong scaling ({proc_count} procs)")
            if not success:
                print(f"\n{Colors.FAIL}✗ Pipeline failed at strong scaling ({proc_count} procs){Colors.ENDC}")
                return 1
    
    # ==================================================================
    # STEP 4: WEAK SCALING EXPERIMENTS
    # ==================================================================
    if run_weak:
        print_section("WEAK SCALING EXPERIMENTS")
        
        for proc_count in args.procs:
            current_step += 1
            print_step(current_step, total_steps, 
                      f"Weak Scaling with {proc_count} processes")
            
            cmd = ['mpirun', '-np', str(proc_count), 
                   'python', 'step2_parallel_knn.py',
                   '--weak-scaling',
                   '--out-dir', args.out_dir]
            
            if args.balanced:
                cmd.append('--balanced')
            
            success = run_command(cmd, 
                                f"Weak scaling ({proc_count} procs)")
            if not success:
                print(f"\n{Colors.FAIL}✗ Pipeline failed at weak scaling ({proc_count} procs){Colors.ENDC}")
                return 1
    
    # ==================================================================
    # STEP 5: ANALYZE AND PLOT RESULTS
    # ==================================================================
    current_step += 1
    print_step(current_step, total_steps, "Analyzing Results")
    
    cmd = ['python', 'step2.1_plot_results.py']
    
    success = run_command(cmd, "Results analysis")
    if not success:
        print(f"\n{Colors.WARNING}⚠ Results analysis failed, but experiments completed{Colors.ENDC}")
    
    # ==================================================================
    # PIPELINE COMPLETE
    # ==================================================================
    print_section("PIPELINE COMPLETE! 🎉")
    
    elapsed = time.time() - pipeline_start
    print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} minutes)\n")
    
    # Print summary
    print(f"{Colors.BOLD}Generated Files:{Colors.ENDC}")
    print(f"  Data:     {args.out_dir}/strong_scaling/")
    if run_weak:
        print(f"            {args.out_dir}/weak_scaling/")
    print(f"  Plots:    ./figures/")
    
    if run_strong:
        print(f"\n{Colors.BOLD}Strong Scaling Results:{Colors.ENDC}")
        print(f"  • strong_speedup_analysis.png")
        print(f"  • strong_efficiency_analysis.png")
        print(f"  • strong_execution_time.png")
        print(f"  • strong_accuracy_analysis.png")
        print(f"  • strong_load_balance_analysis.png")
        print(f"  • strong_performance_summary.csv")
    
    if run_weak:
        print(f"\n{Colors.BOLD}Weak Scaling Results:{Colors.ENDC}")
        print(f"  • weak_weak_scaling_analysis.png")
        print(f"  • weak_execution_time.png")
        print(f"  • weak_performance_summary.csv")
    
    print(f"\n{Colors.OKGREEN}✓ All done! Check the figures/ directory for plots.{Colors.ENDC}\n")
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print(f"\n\n{Colors.WARNING}Pipeline interrupted by user{Colors.ENDC}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Colors.FAIL}Pipeline failed with error:{Colors.ENDC}")
        print(f"{Colors.FAIL}{e}{Colors.ENDC}")
        import traceback
        traceback.print_exc()
        sys.exit(1)