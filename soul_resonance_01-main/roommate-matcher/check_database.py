import sqlite3
import pandas as pd

def check_database():
    """Simple function to check and display database contents"""
    try:
        conn = sqlite3.connect('roommate_submissions.db')
        df = pd.read_sql_query('SELECT * FROM submissions ORDER BY submission_time DESC', conn)
        
        if len(df) > 0:
            print(f"📊 Found {len(df)} submissions:")
            print("=" * 80)
            
            for index, row in df.iterrows():
                print(f"\n🆔 #{row['id']} - {row['name']} ({row['gender']})")
                if 'looking_for' in row and pd.notna(row['looking_for']) and row['looking_for'].strip():
                    print(f"   💭 Looking for: {row['looking_for']}")
                print(f"   Wake-up: {row['wakeup']}")
                print(f"   Sleep: {row['sleep']}")
                print(f"   Study: {row['study_time']}")
                print(f"   Cleanliness: {row['cleanliness']}/5")
                print(f"   Noise tolerance: {row['noise_tolerance']}/5")
                print(f"   Social level: {row['intro_extro']}")
                print(f"   Submitted: {row['submission_time']}")
                print("-" * 50)
        else:
            print("📭 No submissions found in database yet.")
            print("Fill out the form in the Streamlit app to add data!")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    check_database() 