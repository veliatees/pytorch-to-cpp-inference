import urllib.request
import pathlib as pl

path = pl.Path("/Users/veliates/Desktop/Edge AI/dataset/titanic.csv")
path.parent.mkdir(parents=True, exist_ok=True)

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
urllib.request.urlretrieve(url, path)

print(f"İndirildi: {path}")
print(f"Var mı: {path.exists()}")
