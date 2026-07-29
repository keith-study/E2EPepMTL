import argparse
import json
import os
import sys
import time
import numpy as np

from multiprocessing import cpu_count
from E2EPepMTL.e2epepmtl import E2EPepMTL

import os
os.environ["JOBLIB_START_METHOD"] = "spawn"


def parse_cli():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='E2EPepMTL: An End-to-End Automated Machine Learning and Deep Learning Framework Leveraging Protein Language Models for Multi-Task Peptide Property Prediction'
    )

    # Required arguments
    parser.add_argument('positives_dir', type=str, nargs='?', default=None,
                        help='Directory containing positive CSV files (e.g., AB_positives.csv, ACE_positives.csv, etc.), ignored if --benchmark is used')
    parser.add_argument('--outputdir', type=str, default='apdl_result',
                        help='Output directory path, defaults to apdl_result')

    # Basic parameters
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Enable verbose output, defaults to True')
    parser.add_argument('--threads', type=int, default=cpu_count(),
                        help='Number of threads to use, defaults to CPU core count')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed, defaults to 42')

    # Negative sample search parameters
    parser.add_argument('--autosearch', type=str, default='auto',
                        choices=['auto', 'true', 'false'],
                        help='Enable automatic negative sample search (auto/true/false), defaults to auto')
    parser.add_argument('--autosearch_tags', type=str, default='',
                        help='Bioactive tags to exclude from automatic search, comma-separated')
    parser.add_argument('--neg_ratio', type=float, default=1.0,
                        help='Negative to positive sample ratio, float, defaults to 1.0')

    # Data partitioning
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Proportion of data for test set (0-1), defaults to 0.2')
    parser.add_argument('--similarity_threshold', type=float, default=0.3,
                        help='Sequence identity threshold for homology partitioning (0-1), defaults to 0.3')

    # Data embedding
    parser.add_argument('--model_name', type=str, default='esm2_t33_650M_UR50D',
                        choices=[
                            'esm2_t6_8M_UR50D',
                            'esm2_t12_35M_UR50D',
                            'esm2_t30_150M_UR50D',
                            'esm2_t33_650M_UR50D',
                            'esm2_t36_3B_UR50D'
                        ],
                        help='ESM-2 model size. Default: 650M (best performance). '
                             'Use 150M for 5x faster embedding (~2 min vs ~10 min)')

    # Model evaluation
    parser.add_argument('--reuse-data', type=str, default=None,
                        help='Reuse existing preprocessing data (skip Step 1~4, directly train). '
                             'Example: --reuse-data final_data')

    # Single task mode
    parser.add_argument('--single-task', type=str, default=None,
                        help='Run single-task mode for a specific task (e.g., AMP, NEURO). Overrides multi-task mode.')

    return parser.parse_args()


def validate_args(args):
    """Validate command line arguments"""
    if not args.reuse_data:
        if not os.path.isdir(args.positives_dir):
            print(f"Error: Input directory does not exist - {args.positives_dir}")
            sys.exit(1)
    if args.neg_ratio <= 0:
        print("Error: Negative sample ratio must be greater than 0")
        sys.exit(1)
    if not 0 < args.test_size < 1:
        print("Error: Test size must be between 0 and 1")
        sys.exit(1)
    if not 0 <= args.similarity_threshold <= 1:
        print("Error: Similarity threshold must be between 0 and 1")
        sys.exit(1)

def main():
    try:
        args = parse_cli()

        if args.reuse_data:
            args.positives_dir = os.getcwd()
        validate_args(args)

        # Initialize AutoPeptide
        pepmtl = E2EPepMTL(verbose=args.verbose, threads=args.threads, seed=args.seed)

        # Create output directory
        os.makedirs(args.outputdir, exist_ok=True)

        # Save configuration
        config_path = os.path.join(args.outputdir, 'pepmtl_config.json')
        config_dict = vars(args).copy()
        config_dict['note'] = f"Using ESM-2 model: {args.model_name}"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(vars(args), f, indent=4, ensure_ascii=False)

        if args.verbose:
            print("\n=== E2EPepMTL Pipeline Configuration ===")
            print(f"Input positive sample directory: {args.positives_dir}")
            print(f"Output directory: {args.outputdir}")
            print(f"Negative sample ratio: {int(args.neg_ratio)}:1")
            print(f"Mode: {'Single-task' if args.single_task else 'Multi-task'}")
            if args.single_task:
                print(f"Single-task target: {args.single_task}")

        results = pepmtl.run_full_pipeline(
            positives_dir=args.positives_dir,
            outputdir=args.outputdir,
            autosearch=args.autosearch,
            autosearch_tags=args.autosearch_tags,
            neg_ratio=args.neg_ratio,
            test_size=args.test_size,
            similarity_threshold=args.similarity_threshold,
            model_name=args.model_name,
            reuse_data=args.reuse_data,
            task_name=args.single_task
        )

        # Save results
        results_path = os.path.join(args.outputdir, 'pipeline_summary.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'mode': 'single-task' if args.single_task else 'multi-task',
                'task': args.single_task if args.single_task else 'all',
                'results': results
            }, f, indent=4, default=lambda o: int(o) if isinstance(o, np.integer)
                                           else float(o) if isinstance(o, np.floating)
                                           else o)

        if args.verbose:
            print("\n=== Pipeline Results ===")
            print(f"Results saved to: {results_path}")
            print("=" * 80)

    except Exception as e:
        print(f"\nError: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()