import os
import glob
from PyPDF4 import PdfFileReader, PdfFileWriter

print("")
print("     Use this module to unlock the PDFs with the same password for all pdfs")
print("     Like paySlip etc.")
print("")
SourceDir = input('Folder where the protected PDFs are: ')
DestinationDir = input('Folder where the unlocked PDFs should get save: ')
    # C:\Users\ramanathan.hariharan\Documents\SelfLearn\python\OtherStuffs\unlockPdf\New 
password = input('Enter the common password for all files: ')
print("")
employer = input('Are the files are Espire PaySlip (Y/N): ')

if employer.upper()=='Y':
    # Map month to a numerical value (you may need to extend this dictionary for other months)
    month_mapping = {
        'JAN': '01', 'FEB': '02', 'MAR': '03', 'APR': '04', 'MAY': '05', 'JUNE': '06',
        'JULY': '07', 'AUG': '08', 'SEP': '09', 'OCT': '10', 'NOV': '11', 'DEC': '12'
    }

# Use the glob module to create a list of PDF file paths in the directory
pdf_files = glob.glob(os.path.join(SourceDir, '*.pdf'))

if not os.path.exists(DestinationDir):
    # If it doesn't exist, then create the folder
    os.makedirs(DestinationDir)

for pdf_file in pdf_files:
    pdf_filename = os.path.basename(pdf_file)
    new_fileName = pdf_filename
    
    # Create a PdfFileReader object
    pdf_reader = PdfFileReader(open(pdf_file, 'rb'))
    
    # Check if the PDF is encrypted
    if pdf_reader.isEncrypted:
        # Try to decrypt with the provided password
        if pdf_reader.decrypt(password):
            # Create a new PdfFileWriter object
            pdf_writer = PdfFileWriter()
            pdf_writer.addPage(pdf_reader.getPage(0))  # You can add more pages if needed

            if employer.upper()=='Y':
                # Split the file name into parts
                file_parts = pdf_filename.split("'")

                if len(file_parts) == 2:
                    month = file_parts[0]  # Extract the month part (e.g., "Apr")
                    year = file_parts[1][:2]  # Extract the 2 digits after the single quote

                    if month.upper() in month_mapping:
                        month = month_mapping[month.upper()]


                new_fileName = f"20{year}{month}{file_parts[1][2:]}"
                output_path = os.path.join(DestinationDir, new_fileName)
            else:
                # Specify the output path
                output_path = os.path.join(DestinationDir, 'Unlocked_' + pdf_filename)

            # Write the decrypted content to a new PDF file
            with open(output_path, 'wb') as output_file:
                pdf_writer.write(output_file)
        else:
            print(f'Unable to decrypt {pdf_filename} with the provided password.')
    else:
        print(f'{pdf_filename} is not password-protected.')
