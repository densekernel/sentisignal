import subsample, glob

csvs = glob.glob('../data/csv/*.csv')

for csv in csvs:
  print(csv)
  subsample(50, '../data/csv'+csv)



