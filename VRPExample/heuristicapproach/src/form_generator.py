from fpdf import FPDF

# Classe per creare un PDF con layout personalizzato
class DeliveryFormPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 16)
        self.cell(0, 10, "MODULO CONSEGNA", ln=True, align="C")
        self.ln(5)

    def add_delivery_section(self):
        self.set_font("Helvetica", "", 12)
        
        self.cell(0, 10, "Indirizzo di partenza:", ln=True)
        self.rect(self.get_x(), self.get_y(), 180, 10)
        self.ln(15)
        
        self.cell(0, 10, "Indirizzo di destinazione:", ln=True)
        self.rect(self.get_x(), self.get_y(), 180, 10)
        self.ln(15)

        self.cell(0, 10, "Data prevista:", ln=True)
        self.rect(self.get_x(), self.get_y(), 60, 10)
        self.ln(15)

        self.cell(0, 10, "Fascia oraria:", ln=True)
        self.rect(self.get_x(), self.get_y(), 60, 10)
        self.ln(15)

        self.cell(0, 10, "Tempo di servizio (minuti):", ln=True)
        self.rect(self.get_x(), self.get_y(), 40, 10)
        self.ln(15)

        self.cell(0, 10, "Peso carico (kg):", ln=True)
        self.rect(self.get_x(), self.get_y(), 40, 10)
        self.ln(15)

        self.cell(0, 10, "Volume carico (m³):", ln=True)
        self.rect(self.get_x(), self.get_y(), 40, 10)
        self.ln(15)

        self.cell(0, 10, "Numero pallet:", ln=True)
        self.rect(self.get_x(), self.get_y(), 20, 10)
        self.ln(15)

    def add_yes_no_question(self, label):
        self.cell(0, 10, f"{label}:", ln=True)
        x = self.get_x()
        y = self.get_y()

        # "SI" box
        self.rect(x, y, 5, 5)
        self.set_xy(x + 6, y)
        self.cell(10, 5, "SI")

        # "NO" box
        self.set_xy(x + 25, y)
        self.rect(x + 20, y, 5, 5)
        self.set_xy(x + 26, y)
        self.cell(10, 5, "NO")

        self.ln(10)


# Crea e popola il PDF
pdf = DeliveryFormPDF()
pdf.add_page()
pdf.add_delivery_section()
pdf.add_yes_no_question("Trasporto a temperatura controllata?")
pdf.add_yes_no_question("Carico a cura del cliente?")

# Salva il PDF
output_path = "modulo_consegna_vuoto.pdf"
pdf.output(output_path)
