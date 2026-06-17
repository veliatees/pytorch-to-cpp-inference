import pandas as pd
import pathlib as pl

csv_path = pl.Path(__file__).parent / "titanic.csv"
df = pd.read_csv(csv_path)

df= df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)  #drop unnecessary columns
#cabin is taken out because it has too many missing values and is not relevant for our analysis