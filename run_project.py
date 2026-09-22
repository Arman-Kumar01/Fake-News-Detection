"""
Fake News Detection - Project Runner & CLI Interface
Run and evaluate models, test manual held-out dataset, predict custom news,
or launch the web interface.
"""

import os
import sys
import argparse
import pandas as pd
from fake_news_detector import FakeNewsModelSuite, load_dataset, locate_data_file

def print_banner():
    banner = r"""
=============================================================
   ___     _          _  _                  ___       _           _   
  | __|_ _| |_____   | \| |_____ __ _____  |   \ ___ | |_ ___  __| |_ 
  | _/ _` | / / -_)  | .` / -_) V  V (_-<  | |) / -_)|  _/ -_)/ _|  _|
  |_|\__,_|_\_\___|  |_|\_\___|\_/\_//__/  |___/\___| \__\___|\__|\__|
=============================================================
    Machine Learning-Powered Misinformation Detection System
    Algorithms: Logistic Regression | Decision Tree | Random Forest | Gradient Boost
=============================================================
"""
    print(banner)


def run_evaluation(suite):
    """Evaluates the loaded models against the manual_testing.csv dataset."""
    manual_path = locate_data_file("manual_testing.csv")
    print(f"\n[+] Loading Manual Testing Dataset: {manual_path}")
    df_manual = pd.read_csv(manual_path)

    print(f"Total manual test articles: {len(df_manual)}\n")
    print(f"{'Index':<6} | {'Actual Class':<18} | {'Consensus':<18} | {'LR':<10} | {'DT':<10} | {'RF':<10} | {'GB':<10} | {'Match'}")
    print("-" * 95)

    correct_consensus = 0
    total = len(df_manual)

    for idx, row in df_manual.iterrows():
        actual_class = "Not A Fake News" if row.get("class") == 1 else "Fake News"
        text = str(row.get("text", ""))
        pred = suite.predict_article(text)

        consensus = pred["consensus"]
        match = "YES" if consensus == actual_class else "NO"
        if match == "YES":
            correct_consensus += 1

        lr_lbl = pred["models"].get("LR", {}).get("label", "N/A")[:4]
        dt_lbl = pred["models"].get("DT", {}).get("label", "N/A")[:4]
        rf_lbl = pred["models"].get("RF", {}).get("label", "N/A")[:4]
        gb_lbl = pred["models"].get("GB", {}).get("label", "N/A")[:4]

        print(f"{idx:<6} | {actual_class:<18} | {consensus:<18} | {lr_lbl:<10} | {dt_lbl:<10} | {rf_lbl:<10} | {gb_lbl:<10} | {match}")

    accuracy = (correct_consensus / total) * 100
    print("-" * 95)
    print(f"Consensus Accuracy on Manual Test Set: {accuracy:.2f}% ({correct_consensus}/{total} correct)\n")


def display_prediction(pred_result, original_text):
    print("\n" + "=" * 60)
    print("                    PREDICTION RESULTS")
    print("=" * 60)
    print(f"Snippet: \"{original_text[:140]}...\"" if len(original_text) > 140 else f"Snippet: \"{original_text}\"")
    print("-" * 60)
    
    badge = "[ NOT A FAKE NEWS ]" if pred_result["consensus"] == "Not A Fake News" else "[ FAKE NEWS ]"
    print(f"\nFinal Verdict:      {badge}")
    print(f"Confidence:         {pred_result['confidence_percent']}% ({pred_result['confidence_ratio']} models agreed)")
    print(f"Votes:              Fake: {pred_result['votes']['fake']}, Real: {pred_result['votes']['true']}")
    print("\nDetailed Model Breakdown:")
    for k, info in pred_result["models"].items():
        prob_str = ""
        if info["probabilities"]:
            p_fake = info["probabilities"]["fake"] * 100
            p_true = info["probabilities"]["true"] * 100
            prob_str = f" (Confidence: Fake {p_fake:.1f}% / True {p_true:.1f}%)"
        print(f"  * {info['name']:<22}: {info['label']:<16}{prob_str}")
    print("=" * 60 + "\n")


def interactive_mode(suite):
    print("\n[+] Entering Interactive News Testing Mode (Type 'exit' or 'quit' to stop)\n")
    while True:
        try:
            user_input = input("Enter news headline or article text:\n> ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Exiting interactive mode.")
                break
            result = suite.predict_article(user_input)
            display_prediction(result, user_input)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break


def main():
    print_banner()

    parser = argparse.ArgumentParser(description="Fake News Detection System Runner")
    parser.add_argument("--train", action="store_true", help="Force train/retrain the models")
    parser.add_argument("--full", action="store_true", help="Use full 44,000+ article dataset for training")
    parser.add_argument("--samples", type=int, default=10000, help="Number of balanced samples to train on (default: 10000)")
    parser.add_argument("--skip-gb", action="store_true", help="Skip Gradient Boosting for ultrafast training")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate models on manual testing dataset")
    parser.add_argument("--predict", type=str, help="Predict a specific news text string")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive prompt for custom news testing")
    parser.add_argument("--web", action="store_true", help="Launch the web application dashboard")
    parser.add_argument("--port", type=int, default=5000, help="Web server port (default: 5000)")

    args = parser.parse_args()

    if args.web:
        print("[+] Starting Web Dashboard...")
        import app
        app.run_server(port=args.port)
        return

    suite = FakeNewsModelSuite()

    # Determine if training is needed
    should_train = args.train or not suite.is_trained()

    if should_train:
        sample_count = None if args.full else args.samples
        print(f"[+] Starting Model Training Pipeline (Samples: {'Full Dataset' if sample_count is None else sample_count})...")
        df = load_dataset(sample_size=sample_count)
        suite.train(df, skip_gb=args.skip_gb)
    else:
        suite.load()

    if args.predict:
        pred = suite.predict_article(args.predict)
        display_prediction(pred, args.predict)
        return

    if args.interactive:
        interactive_mode(suite)
        return

    # Default action: run evaluation on manual test dataset and run demonstration examples
    print("\n[+] Running Model Evaluation on Held-Out Test Set:")
    run_evaluation(suite)

    # Run Demonstration Examples
    print("[+] Running Sample Predictions:")
    sample_fake = (
        "BREAKING: Pope Francis has endorsed Donald Trump for president in shocking announcement "
        "releasing a statement saying FBI corruption must end immediately."
    )
    sample_real = (
        "WASHINGTON (Reuters) - The U.S. Senate voted on Thursday to approve a bipartisan legislation "
        "aimed at strengthening cybersecurity defenses across federal agencies and critical infrastructure."
    )

    print("\n--- Testing Sample 1 (Expected: Fake News) ---")
    pred1 = suite.predict_article(sample_fake)
    display_prediction(pred1, sample_fake)

    print("--- Testing Sample 2 (Expected: Not A Fake News / Real) ---")
    pred2 = suite.predict_article(sample_real)
    display_prediction(pred2, sample_real)

    print("\n[OK] Project execution completed successfully!")
    print("Tip: Run 'python run_project.py --web' to open the interactive browser dashboard.")
    print("     Run 'python run_project.py --interactive' to paste custom news articles.")


if __name__ == "__main__":
    main()
