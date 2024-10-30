
import glob
import os

import pandas as pd


if __name__ == "__main__":
    # Get all manually-coded files
    manual_files = glob.glob("manual/*.csv")

    valid_matches = []
    for manual_file in manual_files:
        company_name = os.path.basename(manual_file).replace(".csv", "")
        # Load the extracted data as a dataframe
        df = pd.read_csv(f"csv/{company_name}.csv").sort_values(by="year")

        # Load the manual extraction for comparison
        manual_df = pd.read_csv(manual_file).sort_values(by="year")

        print(f"\nResults for {company_name}")
        print("Extracted:")
        print(df.head())
        print("Manual:")
        print(manual_df.head())

        # Compare the two dataframes by checking each value directly
        # First ensure both dataframes have the same columns
        df = df[manual_df.columns]

        # Sort both by year to align rows
        df = df.sort_values("year").reset_index(drop=True)
        manual_df = manual_df.sort_values("year").reset_index(drop=True)

        # Compare all values element by element
        matches = (df == manual_df) | (pd.isna(df) & pd.isna(manual_df))
        is_match = matches.all().all()

        valid_matches.append(is_match)

        if not is_match:
            print("Mismatches:")
            for col in df.columns:
                if not matches[col].all():
                    print(f"\n{col}:")
                    print("Extracted:", df[col].tolist())
                    print("Manual:", manual_df[col].tolist())

    # Report the total matches
    if sum(valid_matches) == len(valid_matches):
        print("\nAll matches are valid!")
    else:
        print(f"\n{sum(valid_matches)}/{len(valid_matches)} matches are valid")
