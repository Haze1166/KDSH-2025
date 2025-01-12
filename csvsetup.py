import csv
def initialise_csv():
    # Open a new file in write mode
    with open('data.csv', 'w', newline='') as csvfile:
        # Create a CSV writer object
        writer = csv.writer(csvfile)
        
        # Write a header row
        writer.writerow(['Publishable', 'File Name', 'Text'])
initialise_csv()