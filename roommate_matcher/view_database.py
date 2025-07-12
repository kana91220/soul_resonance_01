#!/usr/bin/env python3
"""
Simple script to view roommate submission data from SQLite database
"""

import sqlite3
import pandas as pd
from datetime import datetime

def view_submissions():
    """View all submissions from the database"""
    try:
        conn = sqlite3.connect('roommate_submissions.db')
        
        # Get all submissions
        df = pd.read_sql_query("SELECT * FROM submissions ORDER BY submission_time DESC", conn)
        
        if len(df) > 0:
            print(f"\n📊 ROOMMATE SUBMISSIONS DATABASE")
            print(f"{'='*50}")
            print(f"Total submissions: {len(df)}")
            print(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*50}\n")
            
            # Display each submission
            for index, row in df.iterrows():
                print(f"🆔 ID: {row['id']}")
                print(f"👤 Name: {row['name']}")
                print(f"🚻 Gender: {row['gender']}")
                print(f"🌅 Wake-up: {row['wakeup']}")
                print(f"🌙 Sleep: {row['sleep']}")
                print(f"📚 Study Time: {row['study_time']}")
                print(f"🧼 Cleanliness: {row['cleanliness']}/5")
                print(f"🔊 Noise Tolerance: {row['noise_tolerance']}/5")
                print(f"💬 Social Level: {row['intro_extro']}")
                print(f"📅 Submitted: {row['submission_time']}")
                print("-" * 30)
                
        else:
            print("No submissions found in database.")
            
        conn.close()
        
    except sqlite3.Error as e:
        print(f"Database error: {e}")
    except Exception as e:
        print(f"Error: {e}")

def export_to_csv():
    """Export submissions to CSV file"""
    try:
        conn = sqlite3.connect('roommate_submissions.db')
        df = pd.read_sql_query("SELECT * FROM submissions ORDER BY submission_time DESC", conn)
        
        if len(df) > 0:
            filename = f"roommate_submissions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(filename, index=False)
            print(f"✅ Data exported to {filename}")
        else:
            print("No data to export.")
            
        conn.close()
        
    except Exception as e:
        print(f"Export error: {e}")

if __name__ == "__main__":
    print("Roommate Database Viewer")
    print("1. View all submissions")
    print("2. Export to CSV")
    print("3. Exit")
    
    choice = input("\nEnter your choice (1-3): ")
    
    if choice == "1":
        view_submissions()
    elif choice == "2":
        export_to_csv()
    elif choice == "3":
        print("Goodbye!")
    else:
        print("Invalid choice!") 