import re
import nltk
from datetime import datetime
from flask import Flask, render_template, request, send_file, Response
from flask_sqlalchemy import SQLAlchemy
import PyPDF2
import pdfplumber
import fitz
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader, PdfWriter
from fpdf import FPDF
from PIL import Image
from pdf2docx import Converter
import pytesseract
from pdf2image import convert_from_path
from docx import Document
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import io
import zipfile
import pandas as pd
from reportlab.lib.colors import HexColor
import math

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///pdf_operations.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

class PDFFile(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    uploaded_pdf = db.Column(db.String(255), nullable=False)
    manipulated_pdf = db.Column(db.String(255), nullable=True)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

# Create database tables
with app.app_context():
    db.create_all()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/doc')
def doc():
    return render_template('doc.html')

@app.route('/merge', methods=['GET', 'POST'])
def merge():
    if request.method == 'POST':
        files = request.files.getlist('pdfs')
        if len(files) < 2:
            return "Please upload at least two PDF files."
        
        uploaded_names = []
        pdf_buffers = []
        
        for file in files:
            if file.filename == '':
                return "One of the files has no filename."
            if file and file.filename.lower().endswith('.pdf'):
                filename = secure_filename(file.filename)
                uploaded_names.append(filename)
                pdf_buffers.append(io.BytesIO(file.read()))
        
        # Merge PDFs in memory
        output_buffer = io.BytesIO()
        merge_pdfs_in_memory(pdf_buffers, output_buffer)
        output_buffer.seek(0)
        
        # Store in database
        pdf_record = PDFFile(
            uploaded_pdf=','.join(uploaded_names),
            manipulated_pdf='merged.pdf'
        )
        db.session.add(pdf_record)
        db.session.commit()
        
        return send_file(
            output_buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name="merged.pdf"
        )
    return render_template('merge.html')

@app.route('/split', methods=['GET', 'POST'])
def split():
    if request.method == 'POST':
        file = request.files['pdf']
        if file.filename == '':
            return "No file selected."
        
        if file and file.filename.lower().endswith('.pdf'):
            filename = secure_filename(file.filename)
            file_buffer = io.BytesIO(file.read())
            
            # Split the PDF and get a list of in-memory split PDFs
            split_pages = split_pdf_in_memory(file_buffer)
            
            # Create an in-memory ZIP file to store split PDFs
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for pdf_name, pdf_data in split_pages:
                    zip_file.writestr(pdf_name, pdf_data.getvalue())  # Add each PDF to ZIP
            
            # Save to database
            pdf_record = PDFFile(
                uploaded_pdf=filename,
                manipulated_pdf=f"split_{filename}.zip"
            )
            db.session.add(pdf_record)
            db.session.commit()
            
            # Return ZIP file as a response
            zip_buffer.seek(0)
            return send_file(
                zip_buffer,
                mimetype='application/zip',
                as_attachment=True,
                download_name=f"split_{filename}.zip"
            )
    
    return render_template('split.html')

@app.route('/extract-text', methods=['GET', 'POST'])
def extract_text_route():
    if request.method == 'POST':
        file = request.files['pdf']
        if file.filename == '':
            return "No file selected."
        if file and file.filename.lower().endswith('.pdf'):
            filename = secure_filename(file.filename)
            file_buffer = io.BytesIO(file.read())
            
            # Extract text in memory
            text_content = extract_text_in_memory(file_buffer)
            text_buffer = io.BytesIO(text_content.encode('utf-8'))
            
            # Store in database
            pdf_record = PDFFile(
                uploaded_pdf=filename,
                manipulated_pdf=f"{filename}.txt"
            )
            db.session.add(pdf_record)
            db.session.commit()
            
            return send_file(
                text_buffer,
                mimetype='text/plain',
                as_attachment=True,
                download_name=f"{filename}.txt"
            )
    return render_template('extract_text.html')

@app.route('/extract-images', methods=['GET', 'POST'])
def extract_images_route():
    if request.method == 'POST':
        file = request.files['pdf']
        if file.filename == '':
            return "No file selected."
        if file and file.filename.lower().endswith('.pdf'):
            filename = secure_filename(file.filename)
            file_buffer = io.BytesIO(file.read())
            
            # Extract images in memory
            image_buffers = extract_images_in_memory(file_buffer)
            
            if not image_buffers:
                return "No images found in the PDF."
            
            # Create a ZIP file with all extracted images
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for i, img_buffer in enumerate(image_buffers):
                    zip_file.writestr(f"image_{i+1}.png", img_buffer.getvalue())
            
            # Store in database
            pdf_record = PDFFile(
                uploaded_pdf=filename,
                manipulated_pdf=f"images_{filename}.zip"
            )
            db.session.add(pdf_record)
            db.session.commit()
            
            zip_buffer.seek(0)
            return send_file(
                zip_buffer,
                mimetype='application/zip',
                as_attachment=True,
                download_name=f"images_{filename}.zip"
            )
    
    return render_template('extract_images.html')

@app.route('/extract-links', methods=['GET', 'POST'])
def extract_links_route():
    if request.method == 'POST':
        file = request.files['pdf']
        if file.filename == '':
            return "No file selected."
        if file and file.filename.lower().endswith('.pdf'):
            filename = secure_filename(file.filename)
            file_buffer = io.BytesIO(file.read())
            
            # Extract links in memory
            links = extract_links_in_memory(file_buffer)
            links_text = '\n'.join(links)
            links_buffer = io.BytesIO(links_text.encode('utf-8'))
            
            # Store in database
            pdf_record = PDFFile(
                uploaded_pdf=filename,
                manipulated_pdf=f"{filename}_links.txt"
            )
            db.session.add(pdf_record)
            db.session.commit()
            
            return send_file(
                links_buffer,
                mimetype='text/plain',
                as_attachment=True,
                download_name=f"{filename}_links.txt"
            )
    return render_template('extract_links.html')

@app.route('/encrypt', methods=['GET', 'POST'])
def encrypt():
    if request.method == 'POST':
        file = request.files['pdf']
        password = request.form['password']
        if file.filename == '':
            return "No file selected."
        if file and file.filename.lower().endswith('.pdf'):
            filename = secure_filename(file.filename)
            file_buffer = io.BytesIO(file.read())
            
            # Encrypt PDF in memory
            output_buffer = io.BytesIO()
            encrypt_pdf_in_memory(file_buffer, output_buffer, password)
            output_buffer.seek(0)
            
            # Store in database
            pdf_record = PDFFile(
                uploaded_pdf=filename,
                manipulated_pdf=f"encrypted_{filename}"
            )
            db.session.add(pdf_record)
            db.session.commit()
            
            return send_file(
                output_buffer,
                mimetype='application/pdf',
                as_attachment=True,
                download_name=f"encrypted_{filename}"
            )
    return render_template('encrypt.html')

@app.route('/decrypt', methods=['GET', 'POST'])
def decrypt():
    if request.method == 'POST':
        file = request.files['pdf']
        password = request.form['password']
        if file.filename == '':
            return "No file selected."
        if file and file.filename.lower().endswith('.pdf'):
            filename = secure_filename(file.filename)
            file_buffer = io.BytesIO(file.read())
            
            try:
                # Decrypt PDF in memory
                output_buffer = io.BytesIO()
                decrypt_pdf_in_memory(file_buffer, output_buffer, password)
                output_buffer.seek(0)
                
                # Store in database
                pdf_record = PDFFile(
                    uploaded_pdf=filename,
                    manipulated_pdf=f"decrypted_{filename}"
                )
                db.session.add(pdf_record)
                db.session.commit()
                
                return send_file(
                    output_buffer,
                    mimetype='application/pdf',
                    as_attachment=True,
                    download_name=f"decrypted_{filename}"
                )
            except Exception as e:
                return f"Error decrypting PDF: {str(e)}", 400
    return render_template('decrypt.html')

@app.route('/rearrange', methods=['GET', 'POST'])
def rearrange():
    if request.method == 'POST':
        file = request.files['pdf']
        page_order = request.form['page_order']
        if file.filename == '':
            return "No file selected."
        if file and file.filename.lower().endswith('.pdf'):
            filename = secure_filename(file.filename)
            file_buffer = io.BytesIO(file.read())
            
            try:
                # Parse page order
                page_order = [int(p) - 1 for p in page_order.split(',')]
                
                # Rearrange PDF in memory
                output_buffer = io.BytesIO()
                rearrange_pages_in_memory(file_buffer, output_buffer, page_order)
                output_buffer.seek(0)
                
                # Store in database
                pdf_record = PDFFile(
                    uploaded_pdf=filename,
                    manipulated_pdf=f"rearranged_{filename}"
                )
                db.session.add(pdf_record)
                db.session.commit()
                
                return send_file(
                    output_buffer,
                    mimetype='application/pdf',
                    as_attachment=True,
                    download_name=f"rearranged_{filename}"
                )
            except Exception as e:
                return f"Error rearranging PDF: {str(e)}", 400
    return render_template('rearrange.html')

@app.route('/rotate', methods=['GET', 'POST'])
def rotate():
    if request.method == 'POST':
        file = request.files['pdf']
        rotation = int(request.form['rotation'])
        if file.filename == '':
            return "No file selected."
        if file and file.filename.lower().endswith('.pdf'):
            filename = secure_filename(file.filename)
            file_buffer = io.BytesIO(file.read())
            
            # Rotate PDF in memory
            output_buffer = io.BytesIO()
            rotate_pages_in_memory(file_buffer, output_buffer, rotation)
            output_buffer.seek(0)
            
            # Store in database
            pdf_record = PDFFile(
                uploaded_pdf=filename,
                manipulated_pdf=f"rotated_{filename}"
            )
            db.session.add(pdf_record)
            db.session.commit()
            
            return send_file(
                output_buffer,
                mimetype='application/pdf',
                as_attachment=True,
                download_name=f"rotated_{filename}"
            )
    return render_template('rotate.html')

@app.route('/add-metadata', methods=['GET', 'POST'])
def add_metadata_route():
    if request.method == 'POST':
        file = request.files['pdf']
        title = request.form['title']
        author = request.form['author']
        if file.filename == '':
            return "No file selected."
        if file and file.filename.lower().endswith('.pdf'):
            filename = secure_filename(file.filename)
            file_buffer = io.BytesIO(file.read())
            
            # Add metadata in memory
            output_buffer = io.BytesIO()
            add_metadata_in_memory(file_buffer, output_buffer, title, author)
            output_buffer.seek(0)
            
            # Store in database
            pdf_record = PDFFile(
                uploaded_pdf=filename,
                manipulated_pdf=f"metadata_{filename}"
            )
            db.session.add(pdf_record)
            db.session.commit()
            
            return send_file(
                output_buffer,
                mimetype='application/pdf',
                as_attachment=True,
                download_name=f"metadata_{filename}"
            )
    return render_template('add_metadata.html')

@app.route('/read-metadata', methods=['GET', 'POST'])
def read_metadata_route():
    if request.method == 'POST':
        file = request.files['pdf']
        if file.filename == '':
            return "No file selected."
        if file and file.filename.lower().endswith('.pdf'):
            filename = secure_filename(file.filename)
            file_buffer = io.BytesIO(file.read())
            
            # Read metadata in memory
            metadata = read_metadata_in_memory(file_buffer)
            
            # Store in database
            pdf_record = PDFFile(
                uploaded_pdf=filename,
                manipulated_pdf=None  # No manipulation, just reading
            )
            db.session.add(pdf_record)
            db.session.commit()
            
            return render_template('metadata_result.html', metadata=metadata)
    return render_template('read_metadata.html')

@app.route('/create-pdf', methods=['GET', 'POST'])
def create_pdf_route():
    if request.method == 'POST':
        text = request.form.get('text', 'No text provided')
        image = request.files.get('image')
        
        # Create PDF in memory
        output_buffer = io.BytesIO()
        
        if image and image.filename:
            image_buffer = io.BytesIO(image.read())
            create_pdf_in_memory(output_buffer, text, image_buffer)
            image_name = secure_filename(image.filename)
        else:
            create_pdf_in_memory(output_buffer, text)
            image_name = None
        
        output_buffer.seek(0)
        
        # Store in database
        pdf_record = PDFFile(
            uploaded_pdf=image_name if image_name else 'text_only',
            manipulated_pdf='created.pdf'
        )
        db.session.add(pdf_record)
        db.session.commit()
        
        return send_file(
            output_buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name="created.pdf"
        )
    
    return render_template('create_pdf.html')

@app.route('/optimize', methods=['GET', 'POST'])
def optimize():
    if request.method == 'POST':
        file = request.files['pdf']
        if file.filename == '':
            return "No file selected."
        if file and file.filename.lower().endswith('.pdf'):
            filename = secure_filename(file.filename)
            file_buffer = io.BytesIO(file.read())
            
            # Optimize PDF in memory
            output_buffer = io.BytesIO()
            optimize_pdf_in_memory(file_buffer, output_buffer)
            output_buffer.seek(0)
            
            # Store in database
            pdf_record = PDFFile(
                uploaded_pdf=filename,
                manipulated_pdf=f"optimized_{filename}"
            )
            db.session.add(pdf_record)
            db.session.commit()
            
            return send_file(
                output_buffer,
                mimetype='application/pdf',
                as_attachment=True,
                download_name=f"optimized_{filename}"
            )
    return render_template('optimize.html')

@app.route('/convert', methods=['GET', 'POST'])
def convert_pdf_to_word():
    if request.method == 'POST':
        if 'pdf_file' not in request.files:
            return "No file uploaded", 400
        
        file = request.files['pdf_file']
        if file.filename == '':
            return "No selected file", 400
        
        if file and file.filename.lower().endswith('.pdf'):
            pdf_filename = secure_filename(file.filename)
            file_buffer = io.BytesIO(file.read())
            
            try:
                # Convert PDF to Word in memory
                docx_buffer = io.BytesIO()
                convert_pdf_to_word_in_memory(file_buffer, docx_buffer)
                docx_buffer.seek(0)
                
                # Store in database
                pdf_record = PDFFile(
                    uploaded_pdf=pdf_filename,
                    manipulated_pdf=f"{pdf_filename.rsplit('.', 1)[0]}.docx"
                )
                db.session.add(pdf_record)
                db.session.commit()
                
                return send_file(
                    docx_buffer,
                    mimetype='application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                    as_attachment=True,
                    download_name=f"{pdf_filename.rsplit('.', 1)[0]}.docx"
                )
            except Exception as e:
                return f"Error during conversion: {str(e)}", 500
        
        return "Invalid file format. Please upload a PDF.", 400
    
    return render_template('convert_pdf_to_word.html')

@app.route('/rename-pdf', methods=['GET', 'POST'])
def rename_pdf_route():
    if request.method == 'POST':
        file = request.files.get('pdf')
        if not file or file.filename == '':
            return "No file selected."
        
        new_name = request.form.get('new_name', '')
        if not new_name:
            return "No new name provided."
        
        if not file.filename.lower().endswith('.pdf'):
            return "Invalid file type. Only PDFs are allowed."
        
        # Ensure new name has .pdf extension
        if not new_name.lower().endswith('.pdf'):
            new_name += '.pdf'
        
        new_name = secure_filename(new_name)
        
        # Just read the file into memory
        pdf_buffer = io.BytesIO(file.read())
        pdf_buffer.seek(0)
        
        # Store in database
        pdf_record = PDFFile(
            uploaded_pdf=secure_filename(file.filename),
            manipulated_pdf=new_name
        )
        db.session.add(pdf_record)
        db.session.commit()
        
        return send_file(
            pdf_buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=new_name
        )
    
    return render_template('rename_pdf.html')

@app.route("/convert-to-excel", methods=["GET", "POST"])
def convert_to_excel():
    if request.method == "POST":
        try:
            if "pdf" not in request.files:
                return "No file uploaded", 400
            
            file = request.files["pdf"]
            if file.filename == "" or not allowed_file(file.filename):
                return "Invalid file type. Only PDFs allowed.", 400
            
            filename = secure_filename(file.filename)
            file_buffer = io.BytesIO(file.read())
            
            # Convert PDF to Excel in memory
            excel_buffer = io.BytesIO()
            convert_pdf_to_excel_in_memory(file_buffer, excel_buffer)
            excel_buffer.seek(0)
            
            # Store in database
            pdf_record = PDFFile(
                uploaded_pdf=filename,
                manipulated_pdf=f"{filename.rsplit('.', 1)[0]}.xlsx"
            )
            db.session.add(pdf_record)
            db.session.commit()
            
            return send_file(
                excel_buffer,
                mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                as_attachment=True,
                download_name=f"{filename.rsplit('.', 1)[0]}.xlsx"
            )
        
        except Exception as e:
            return f"Error during conversion: {str(e)}", 500
    
    return render_template("convert_to_excel.html")

# PDF manipulation functions (in-memory versions)
def merge_pdfs_in_memory(pdf_buffers, output_buffer):
    pdf_writer = PyPDF2.PdfWriter()
    
    for pdf_buffer in pdf_buffers:
        pdf_reader = PyPDF2.PdfReader(pdf_buffer)
        for page_num in range(len(pdf_reader.pages)):
            pdf_writer.add_page(pdf_reader.pages[page_num])
    
    pdf_writer.write(output_buffer)

def split_pdf_in_memory(pdf_buffer):
    pdf_reader = PdfReader(pdf_buffer)
    split_files = []
    
    for i, page in enumerate(pdf_reader.pages):
        pdf_writer = PdfWriter()
        pdf_writer.add_page(page)
        
        # Create an in-memory buffer for the split page
        page_buffer = io.BytesIO()
        pdf_writer.write(page_buffer)
        page_buffer.seek(0)
        
        split_files.append((f"split_page_{i+1}.pdf", page_buffer))
    
    return split_files

def extract_text_in_memory(pdf_buffer):
    with pdfplumber.open(pdf_buffer) as pdf:
        full_text = ''
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text += text + '\n'
    return full_text

def extract_images_in_memory(pdf_buffer):
    doc = fitz.open(stream=pdf_buffer.read(), filetype="pdf")
    image_buffers = []
    
    for page_num in range(len(doc)):
        for img_index, img in enumerate(doc[page_num].get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            
            # Store image in memory
            img_buffer = io.BytesIO(image_bytes)
            image_buffers.append(img_buffer)
    
    return image_buffers

def extract_links_in_memory(pdf_buffer):
    pdf_document = fitz.open(stream=pdf_buffer.read(), filetype="pdf")
    links = []
    
    for page_index in range(len(pdf_document)):
        page = pdf_document.load_page(page_index)
        link_list = page.get_links()
        for link in link_list:
            if link.get("uri"):
                links.append(link["uri"])
    
    return links

def encrypt_pdf_in_memory(input_buffer, output_buffer, password):
    pdf_reader = PyPDF2.PdfReader(input_buffer)
    pdf_writer = PyPDF2.PdfWriter()
    
    for page_num in range(len(pdf_reader.pages)):
        pdf_writer.add_page(pdf_reader.pages[page_num])
    
    pdf_writer.encrypt(password)
    pdf_writer.write(output_buffer)

def decrypt_pdf_in_memory(input_buffer, output_buffer, password):
    pdf_reader = PyPDF2.PdfReader(input_buffer)
    pdf_reader.decrypt(password)
    
    pdf_writer = PyPDF2.PdfWriter()
    for page_num in range(len(pdf_reader.pages)):
        pdf_writer.add_page(pdf_reader.pages[page_num])
    
    pdf_writer.write(output_buffer)

def rearrange_pages_in_memory(input_buffer, output_buffer, page_order):
    pdf_reader = PyPDF2.PdfReader(input_buffer)
    pdf_writer = PyPDF2.PdfWriter()
    
    for page_num in page_order:
        if 0 <= page_num < len(pdf_reader.pages):
            pdf_writer.add_page(pdf_reader.pages[page_num])
    
    pdf_writer.write(output_buffer)

def rotate_pages_in_memory(input_buffer, output_buffer, rotation):
    pdf_reader = PyPDF2.PdfReader(input_buffer)
    pdf_writer = PyPDF2.PdfWriter()
    
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        page.rotate(rotation)
        pdf_writer.add_page(page)
    
    pdf_writer.write(output_buffer)

def add_metadata_in_memory(input_buffer, output_buffer, title, author):
    pdf_reader = PyPDF2.PdfReader(input_buffer)
    pdf_writer = PyPDF2.PdfWriter()
    
    for page_num in range(len(pdf_reader.pages)):
        pdf_writer.add_page(pdf_reader.pages[page_num])
    
    metadata = {
        '/Title': title,
        '/Author': author,
        '/Producer': ''
    }
    pdf_writer.add_metadata(metadata)
    pdf_writer.write(output_buffer)

def read_metadata_in_memory(pdf_buffer):
    pdf_reader = PyPDF2.PdfReader(pdf_buffer)
    return pdf_reader.metadata

def create_pdf_in_memory(output_buffer, text, image_buffer=None):
    c = canvas.Canvas(output_buffer, pagesize=letter)
    
    # Add user text
    c.drawString(100, 750, text)
    
    # Add image if provided
    if image_buffer:
        try:
            img = Image.open(image_buffer)
            img_buffer = io.BytesIO()
            img.save(img_buffer, format=img.format)
            img_buffer.seek(0)
            
            c.drawImage(img_buffer, 100, 500, width=200, height=150)
        except Exception as e:
            print(f"Error adding image: {e}")
    
    c.save()

def optimize_pdf_in_memory(input_buffer, output_buffer):
    doc = fitz.open(stream=input_buffer.read(), filetype="pdf")
    doc.save(output_buffer, garbage=3, deflate=True)

def convert_pdf_to_word_in_memory(pdf_buffer, docx_buffer):
    # Create temporary files since pdf2docx requires file paths
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_pdf:
        temp_pdf.write(pdf_buffer.read())
        temp_pdf_path = temp_pdf.name
    
    with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as temp_docx:
        temp_docx_path = temp_docx.name
    
    try:
        # Convert PDF to Word
        cv = Converter(temp_pdf_path)
        cv.convert(temp_docx_path, start=0, end=None)
        cv.close()
        
        # Read the output file into the buffer
        with open(temp_docx_path, 'rb') as f:
            docx_buffer.write(f.read())
    
    finally:
        # Clean up temporary files
        if os.path.exists(temp_pdf_path):
            os.unlink(temp_pdf_path)
        if os.path.exists(temp_docx_path):
            os.unlink(temp_docx_path)

def convert_pdf_to_excel_in_memory(pdf_buffer, excel_buffer):
    # Extract tables from PDF
    tables_df = extract_tables_from_buffer(pdf_buffer)
    text_df = extract_text_from_buffer(pdf_buffer)
    invoice_df = extract_invoice_data_from_buffer(pdf_buffer)
    summary_df = summarize_pdf_from_buffer(pdf_buffer)
    
    # Write to Excel
    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
        if tables_df is not None and not tables_df.empty:
            tables_df.to_excel(writer, sheet_name="Tables", index=False)
        
        if not text_df.empty:
            text_df.to_excel(writer, sheet_name="Text Content", index=False)
        
        if not invoice_df.empty:
            invoice_df.to_excel(writer, sheet_name="Invoice Details", index=False)
        
        if not summary_df.empty:
            summary_df.to_excel(writer, sheet_name="Summary", index=False)

def extract_tables_from_buffer(pdf_buffer):
    try:
        all_tables = []
        with pdfplumber.open(pdf_buffer) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                tables = page.extract_tables()
                if tables:
                    for table_num, table in enumerate(tables, 1):
                        cleaned_table = []
                        for row in table:
                            cleaned_row = [str(cell).strip() if cell else '' for cell in row]
                            cleaned_table.append(cleaned_row)
                        
                        df = pd.DataFrame(cleaned_table)
                        if not df.empty:
                            if df.iloc[0].astype(str).str.match(r'^[A-Za-z]').all():
                                df.columns = df.iloc[0]
                                df = df.iloc[1:]
                            
                            df.insert(0, 'Page', page_num)
                            df.insert(1, 'Table', table_num)
                            all_tables.append(df)
        
        if all_tables:
            final_df = pd.concat(all_tables, ignore_index=True)
            return final_df
        return pd.DataFrame()
    
    except Exception as e:
        print(f"Error in table extraction: {str(e)}")
        return pd.DataFrame()

def extract_text_from_buffer(pdf_buffer):
    try:
        pdf_buffer.seek(0)
        doc = fitz.open(stream=pdf_buffer.read(), filetype="pdf")
        text_data = []
        
        for page_num, page in enumerate(doc, 1):
            blocks = page.get_text("blocks")
            
            for block in blocks:
                x0, y0, x1, y1, text, block_num, block_type = block
                
                text = text.strip()
                if text:
                    is_header = False
                    words = page.get_text("words")
                    for word in words:
                        if word[4] in text:
                            font_size = word[5]
                            if font_size > 12:
                                is_header = True
                                break
                    
                    text_data.append({
                        'Page': page_num,
                        'Type': 'Header' if is_header else 'Content',
                        'Position': f"y: {int(y0)}",
                        'Content': text
                    })
        
        df = pd.DataFrame(text_data)
        if not df.empty:
            df = df.sort_values(['Page', 'Position'])
        return df
    
    except Exception as e:
        print(f"Error in text extraction: {str(e)}")
        return pd.DataFrame()

def extract_invoice_data_from_buffer(pdf_buffer):
    try:
        pdf_buffer.seek(0)
        with pdfplumber.open(pdf_buffer) as pdf:
            text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
        
        patterns = {
            'Invoice Number': [
                r'Invoice\s*(?:No|Number|#)[:.]?\s*([A-Za-z0-9\-]+)',
                r'Invoice:\s*([A-Za-z0-9\-]+)'
            ],
            'Date': [
                r'(?:Invoice )?Date[:.]?\s*([\d]{1,2}[-/][\d]{1,2}[-/][\d]{2,4})',
                r'Date:\s*(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{2,4})'
            ],
            'Due Date': [
                r'Due\s*Date[:.]?\s*([\d]{1,2}[-/][\d]{1,2}[-/][\d]{2,4})'
            ],
            'Total Amount': [
                r'(?:Total|Amount|Sum)[:.]?\s*[\$£€]?([\d,]+\.?\d*)',
                r'Total\s*Due[:.]?\s*[\$£€]?([\d,]+\.?\d*)'
            ],
            'Tax Amount': [
                r'(?:Tax|VAT)[:.]?\s*[\$£€]?([\d,]+\.?\d*)'
            ],
            'Company Name': [
                r'(?:From|Company|Vendor)[:.]?\s*([A-Za-z0-9\s,]+(?:Inc|LLC|Ltd|Limited|Corp|Corporation)?)',
                r'^([A-Za-z0-9\s,]+(?:Inc|LLC|Ltd|Limited|Corp|Corporation)?)'
            ]
        }
        
        extracted_data = {}
        for field, pattern_list in patterns.items():
            for pattern in pattern_list:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    extracted_data[field] = match.group(1).strip()
                    break
            if field not in extracted_data:
                extracted_data[field] = "N/A"
        
        return pd.DataFrame([extracted_data])
    
    except Exception as e:
        print(f"Error in invoice data extraction: {str(e)}")
        return pd.DataFrame()

def summarize_pdf_from_buffer(pdf_buffer, sentence_count=5):
    try:
        pdf_buffer.seek(0)
        doc = fitz.open(stream=pdf_buffer.read(), filetype="pdf")
        
        text_blocks = []
        for page in doc:
            blocks = page.get_text("blocks")
            for block in blocks:
                text = block[4].strip()
                if len(text) > 50:
                    text_blocks.append(text)
        
        text = "\n".join(text_blocks)
        
        if not text.strip():
            return pd.DataFrame([{"Summary": "No substantial text found in PDF"}])
        
        sentences = nltk.sent_tokenize(text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        if not sentences:
            return pd.DataFrame([{"Summary": "No meaningful sentences found for summarization"}])
        
        parser = PlaintextParser.from_string(" ".join(sentences), Tokenizer("english"))
        summarizer = LsaSummarizer()
        summary = summarizer(parser.document, sentence_count)
        
        summary_text = "\n• " + "\n• ".join(str(sentence) for sentence in summary)
        
        return pd.DataFrame([{
            "Document Length": f"{len(sentences)} sentences",
            "Summary": summary_text
        }])
    
    except Exception as e:
        print(f"Error in PDF summarization: {str(e)}")
        return pd.DataFrame([{"Summary": f"Error creating summary: {str(e)}"}])

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in {"pdf"}

# Add missing imports
import tempfile
import os

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run
