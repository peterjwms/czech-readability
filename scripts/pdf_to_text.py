# Attempting to use the pdfminer, get an AttributeError(no attribute seek for str object) on line 17

from glob import glob
import pdfminer
import io

from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfpage import PDFPage

pdf_dir = "C:\\Users\\peter\\Repos\\czech_readability_corpus\\orig\\"
counter = 0
for fname in glob(pdf_dir + "unlocked*.pdf"):
    print(fname)

    for page in PDFPage.get_pages(fname):
        resource_manager = PDFResourceManager()
        fake_file_handle = io.StringIO()

        converter = TextConverter(resource_manager)
        page_interpreter = PDFPageInterpreter(resource_manager, converter)

        page_interpreter.process_page(page)
        text = fake_file_handle.getvalue()
        print(text)
        break

    break

