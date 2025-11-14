import os 
import data_loader 

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "data", "players.csv")

    df = data_loader.load_csv("data/players.csv")
    print("CVS chargé avec succès.")

if __name__ == "__main__":
    main()