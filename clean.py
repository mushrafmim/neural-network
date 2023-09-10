import csv

with open('task_1/true-dw.csv') as dw:
    csv_reader = csv.reader(dw)

    for row in csv_reader:
        print(len(row))

print("\n\n\n")

with open('task_1/w.csv') as w:
    csv_reader = csv.reader(w)

    for row in csv_reader:

        if csv_reader.line_num <= 14:
            print(row[1:])
