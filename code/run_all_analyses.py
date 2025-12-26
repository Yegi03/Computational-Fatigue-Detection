#!/usr/bin/env python3
"""
Run All Research Question Analyses
Execute all RQ analyses and save results
"""

import json
from pathlib import Path

# Import RQ analysis functions
from run_rq1_pfi_accuracy import run_rq1_analysis
from run_rq2_temporal_ceiling import run_rq2_analysis
from run_rq3_modality_ranking import run_rq3_analysis

def run_all_analyses():
    """Run all research question analyses"""
    print("=" * 60)
    print("RUNNING ALL RESEARCH QUESTION ANALYSES")
    print("=" * 60)
    
    results = {}
    
    # Run RQ1
    print("\n")
    results['rq1'] = run_rq1_analysis()
    
    # Run RQ2
    print("\n")
    results['rq2'] = run_rq2_analysis()
    
    # Run RQ3
    print("\n")
    results['rq3'] = run_rq3_analysis()
    
    # Save combined results
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "all_rq_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("ALL ANALYSES COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {output_dir / 'all_rq_results.json'}")
    
    return results

if __name__ == "__main__":
    run_all_analyses()

