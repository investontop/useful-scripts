import os
import glob
from PyPDF4 import PdfFileReader, PdfFileWriter

print("")
print("     Use this module to unlock the PDFs with the same password for all pdfs")
print("     Like paySlip etc.")
print("")
SourceDir = input('Folder where the protected PDFs are: ')
DestinationDir = input('Folder where the unlocked PDFs should get save: ')
password = input('Enter the common password for all files: ')

# Use the glob module to create a list of PDF file paths in the directory
pdf_files = glob.glob(os.path.join(SourceDir, '*.pdf'))

for pdf_file in pdf_files:
    pdf_filename = os.path.basename(pdf_file)
    
    # Create a PdfFileReader object
    pdf_reader = PdfFileReader(open(pdf_file, 'rb'))
    
    # Check if the PDF is encrypted
    if pdf_reader.isEncrypted:
        # Try to decrypt with the provided password
        if pdf_reader.decrypt(password):
            # Create a new PdfFileWriter object
            pdf_writer = PdfFileWriter()
            pdf_writer.addPage(pdf_reader.getPage(0))  # You can add more pages if needed

            # Specify the output path
            output_path = os.path.join(DestinationDir, 'Unlocked_' + pdf_filename)

            # Write the decrypted content to a new PDF file
            with open(output_path, 'wb') as output_file:
                pdf_writer.write(output_file)
        else:
            print(f'Unable to decrypt {pdf_filename} with the provided password.')
    else:
        print(f'{pdf_filename} is not password-protected.')
