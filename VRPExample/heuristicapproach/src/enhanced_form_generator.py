"""
Enhanced Form Generator for VRP Data Collection

This module generates professional PDF forms for collecting VRP-related data
from clients, eliminating the need for Excel files and enabling direct 
database integration.

Features:
- Multiple form types (Orders, Vehicles, Drivers)
- Professional layout and formatting
- Database integration ready
- Multi-language support (Italian/English)
- QR codes for digital processing
"""

from fpdf import FPDF
import qrcode
from datetime import datetime
from typing import List, Dict, Optional, Any
import json
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class FormConfig:
    """Configuration for form generation."""
    language: str = "it"  # "it" for Italian, "en" for English
    company_name: str = "Optimize Logistics"
    company_logo: Optional[str] = None
    output_directory: str = "forms"
    include_qr_code: bool = True
    form_id_prefix: str = "VRP"


class VRPFormGenerator(FPDF):
    """Enhanced PDF form generator for VRP data collection."""
    
    def __init__(self, config: FormConfig = None):
        super().__init__()
        self.config = config or FormConfig()
        self.form_id = f"{self.config.form_id_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Text translations
        self.texts = self._load_translations()
        
    def _load_translations(self) -> Dict[str, Dict[str, str]]:
        """Load text translations for different languages."""
        return {
            "it": {
                "delivery_form": "MODULO ORDINE CONSEGNA",
                "vehicle_form": "MODULO INFORMAZIONI VEICOLO", 
                "driver_form": "MODULO INFORMAZIONI AUTISTA",
                "pickup_address": "Indirizzo di ritiro:",
                "delivery_address": "Indirizzo di consegna:",
                "expected_date": "Data prevista:",
                "time_window": "Fascia oraria:",
                "service_time": "Tempo di servizio (minuti):",
                "cargo_weight": "Peso carico (kg):",
                "cargo_volume": "Volume carico (m³):",
                "pallets_count": "Numero pallet:",
                "special_requirements": "Requisiti speciali:",
                "temperature_controlled": "Trasporto a temperatura controllata?",
                "customer_loading": "Carico a cura del cliente?",
                "fragile_goods": "Merci fragili?",
                "hazmat": "Materiali pericolosi?",
                "priority": "Priorità (1-5):",
                "contact_person": "Persona di contatto:",
                "phone_number": "Numero di telefono:",
                "email": "Email:",
                "notes": "Note aggiuntive:",
                "yes": "SÌ",
                "no": "NO",
                "form_id": "ID Modulo:",
                "date_compiled": "Data compilazione:",
                "signature": "Firma:",
                # Vehicle form
                "vehicle_plate": "Targa veicolo:",
                "vehicle_type": "Tipo veicolo:",
                "max_weight": "Peso massimo (kg):",
                "max_volume": "Volume massimo (m³):",
                "max_pallets": "Pallet massimi:",
                "fuel_type": "Tipo carburante:",
                "emission_class": "Classe emissioni:",
                "special_equipment": "Attrezzature speciali:",
                "refrigeration": "Refrigerazione",
                "crane": "Gru",
                "lift_gate": "Sponda idraulica",
                "gps_tracker": "Tracker GPS",
                # Driver form
                "driver_name": "Nome autista:",
                "license_type": "Tipo patente:",
                "license_expiry": "Scadenza patente:",
                "experience_years": "Anni di esperienza:",
                "certifications": "Certificazioni:",
                "adr_license": "Patente ADR",
                "forklift_license": "Patente muletto",
                "preferred_routes": "Rotte preferite:",
                "availability": "Disponibilità:",
                "cost_per_hour": "Costo orario (EUR):",
            },
            "en": {
                "delivery_form": "DELIVERY ORDER FORM",
                "vehicle_form": "VEHICLE INFORMATION FORM",
                "driver_form": "DRIVER INFORMATION FORM", 
                "pickup_address": "Pickup address:",
                "delivery_address": "Delivery address:",
                "expected_date": "Expected date:",
                "time_window": "Time window:",
                "service_time": "Service time (minutes):",
                "cargo_weight": "Cargo weight (kg):",
                "cargo_volume": "Cargo volume (m³):",
                "pallets_count": "Number of pallets:",
                "special_requirements": "Special requirements:",
                "temperature_controlled": "Temperature controlled transport?",
                "customer_loading": "Customer loading?",
                "fragile_goods": "Fragile goods?",
                "hazmat": "Hazardous materials?",
                "priority": "Priority (1-5):",
                "contact_person": "Contact person:",
                "phone_number": "Phone number:",
                "email": "Email:",
                "notes": "Additional notes:",
                "yes": "YES",
                "no": "NO",
                "form_id": "Form ID:",
                "date_compiled": "Date compiled:",
                "signature": "Signature:",
                # Vehicle form
                "vehicle_plate": "Vehicle plate:",
                "vehicle_type": "Vehicle type:",
                "max_weight": "Max weight (kg):",
                "max_volume": "Max volume (m³):",
                "max_pallets": "Max pallets:",
                "fuel_type": "Fuel type:",
                "emission_class": "Emission class:",
                "special_equipment": "Special equipment:",
                "refrigeration": "Refrigeration",
                "crane": "Crane",
                "lift_gate": "Lift gate",
                "gps_tracker": "GPS tracker",
                # Driver form
                "driver_name": "Driver name:",
                "license_type": "License type:",
                "license_expiry": "License expiry:",
                "experience_years": "Years of experience:",
                "certifications": "Certifications:",
                "adr_license": "ADR license",
                "forklift_license": "Forklift license", 
                "preferred_routes": "Preferred routes:",
                "availability": "Availability:",
                "cost_per_hour": "Cost per hour (EUR):",
            }
        }
    
    def get_text(self, key: str) -> str:
        """Get translated text for the current language."""
        return self.texts.get(self.config.language, self.texts["en"]).get(key, key)
    
    def header(self):
        """Override FPDF header method."""
        pass  # We'll call our custom header method manually
    
    def add_form_header(self, title: str):
        """Add header with company info and form title."""
        # Company name
        self.set_font("Helvetica", "B", 20)
        self.cell(0, 10, self.config.company_name, ln=True, align="C")
        self.ln(2)
        
        # Form title
        self.set_font("Helvetica", "B", 16)
        self.cell(0, 10, title, ln=True, align="C")
        self.ln(5)
        
        # Form metadata
        self.set_font("Helvetica", "", 10)
        date_str = datetime.now().strftime("%d/%m/%Y %H:%M")
        
        # Form ID and date in two columns
        self.cell(95, 8, f"{self.get_text('form_id')} {self.form_id}", border=1)
        self.cell(95, 8, f"{self.get_text('date_compiled')} {date_str}", border=1, ln=True)
        self.ln(10)
    
    def add_text_field(self, label: str, width: int = 120, required: bool = True):
        """Add a labeled text input field."""
        self.set_font("Helvetica", "", 11)
        
        # Add asterisk for required fields
        marker = " *" if required else ""
        self.cell(60, 8, f"{label}{marker}", align="L")
        
        # Input box
        self.rect(self.get_x(), self.get_y(), width, 8)
        self.ln(12)
    
    def add_number_field(self, label: str, width: int = 60, required: bool = True):
        """Add a labeled number input field."""
        self.add_text_field(label, width, required)
    
    def add_yes_no_question(self, label: str, required: bool = True):
        """Add a yes/no checkbox question."""
        self.set_font("Helvetica", "", 11)
        
        # Add asterisk for required fields
        marker = " *" if required else ""
        self.cell(0, 8, f"{label}{marker}", ln=True)
        
        x = self.get_x()
        y = self.get_y()
        
        # YES checkbox
        self.rect(x + 10, y, 5, 5)
        self.set_xy(x + 17, y + 1)
        self.set_font("Helvetica", "", 10)
        self.cell(20, 5, self.get_text("yes"))
        
        # NO checkbox  
        self.rect(x + 50, y, 5, 5)
        self.set_xy(x + 57, y + 1)
        self.cell(20, 5, self.get_text("no"))
        
        self.ln(10)
    
    def add_checkbox_group(self, label: str, options: List[str], required: bool = False):
        """Add a group of checkboxes."""
        self.set_font("Helvetica", "", 11)
        
        marker = " *" if required else ""
        self.cell(0, 8, f"{label}{marker}", ln=True)
        
        x_start = self.get_x() + 10
        y_start = self.get_y()
        
        for i, option in enumerate(options):
            x = x_start
            y = y_start + (i * 7)
            
            self.rect(x, y, 4, 4)
            self.set_xy(x + 6, y)
            self.set_font("Helvetica", "", 10)
            self.cell(0, 4, option)
        
        self.ln(len(options) * 7 + 5)
    
    def add_large_text_area(self, label: str, height: int = 30, required: bool = False):
        """Add a large text area for notes/comments."""
        self.set_font("Helvetica", "", 11)
        
        marker = " *" if required else ""
        self.cell(0, 8, f"{label}{marker}", ln=True)
        
        # Large text box
        self.rect(self.get_x(), self.get_y(), 180, height)
        self.ln(height + 5)
    
    def add_signature_section(self):
        """Add signature section."""
        self.ln(10)
        self.set_font("Helvetica", "B", 12)
        self.cell(0, 8, self.get_text("signature"), ln=True)
        
        # Signature line
        y = self.get_y() + 20
        self.line(20, y, 100, y)
        self.set_xy(20, y + 2)
        self.set_font("Helvetica", "", 9)
        self.cell(80, 5, "Firma del cliente / Customer signature", align="C")
        
        # Date line
        self.line(120, y, 180, y)
        self.set_xy(120, y + 2)
        self.cell(60, 5, "Data / Date", align="C")
    
    def add_qr_code(self, data: Dict[str, Any]):
        """Add QR code with form data for digital processing."""
        if not self.config.include_qr_code:
            return
            
        # Create QR code
        qr_data = json.dumps(data)
        qr = qrcode.QRCode(version=1, box_size=3, border=1)
        qr.add_data(qr_data)
        qr.make(fit=True)
        
        # Save QR code as temporary image
        qr_img = qr.make_image(fill_color="black", back_color="white")
        qr_path = f"temp_qr_{self.form_id}.png"
        qr_img.save(qr_path)
        
        # Add QR code to PDF
        self.set_xy(150, 250)
        try:
            self.image(qr_path, w=30, h=30)
            self.set_xy(150, 285)
            self.set_font("Helvetica", "", 8)
            self.cell(30, 4, "Scan for digital", align="C", ln=True)
            self.set_x(150)
            self.cell(30, 4, "processing", align="C")
        finally:
            # Clean up temporary file
            if os.path.exists(qr_path):
                os.remove(qr_path)
    
    def generate_delivery_order_form(self) -> str:
        """Generate delivery order form."""
        self.add_page()
        self.add_form_header(self.get_text("delivery_form"))
        
        # Basic order information
        self.add_text_field(self.get_text("pickup_address"), 150)
        self.add_text_field(self.get_text("delivery_address"), 150)
        self.add_text_field(self.get_text("expected_date"), 80)
        self.add_text_field(self.get_text("time_window"), 100)
        
        # Cargo details
        self.add_number_field(self.get_text("cargo_weight"), 60)
        self.add_number_field(self.get_text("cargo_volume"), 60)
        self.add_number_field(self.get_text("pallets_count"), 40)
        self.add_number_field(self.get_text("service_time"), 60)
        self.add_number_field(self.get_text("priority"), 40)
        
        # Special requirements
        self.add_yes_no_question(self.get_text("temperature_controlled"))
        self.add_yes_no_question(self.get_text("customer_loading"))
        self.add_yes_no_question(self.get_text("fragile_goods"))
        self.add_yes_no_question(self.get_text("hazmat"))
        
        # Contact information
        self.add_text_field(self.get_text("contact_person"), 120)
        self.add_text_field(self.get_text("phone_number"), 100)
        self.add_text_field(self.get_text("email"), 120)
        
        # Notes
        self.add_large_text_area(self.get_text("notes"))
        
        # Signature
        self.add_signature_section()
        
        # QR Code
        qr_data = {
            "form_type": "delivery_order",
            "form_id": self.form_id,
            "generated_at": datetime.now().isoformat()
        }
        self.add_qr_code(qr_data)
        
        return self.form_id
    
    def generate_vehicle_form(self) -> str:
        """Generate vehicle information form."""
        self.add_page()
        self.add_form_header(self.get_text("vehicle_form"))
        
        # Vehicle identification
        self.add_text_field(self.get_text("vehicle_plate"), 100)
        self.add_text_field(self.get_text("vehicle_type"), 120)
        
        # Capacity specifications
        self.add_number_field(self.get_text("max_weight"), 80)
        self.add_number_field(self.get_text("max_volume"), 80)
        self.add_number_field(self.get_text("max_pallets"), 60)
        
        # Technical specifications
        self.add_text_field(self.get_text("fuel_type"), 100)
        self.add_text_field(self.get_text("emission_class"), 80)
        
        # Special equipment
        equipment_options = [
            self.get_text("refrigeration"),
            self.get_text("crane"),
            self.get_text("lift_gate"),
            self.get_text("gps_tracker")
        ]
        self.add_checkbox_group(self.get_text("special_equipment"), equipment_options)
        
        # Notes
        self.add_large_text_area(self.get_text("notes"))
        
        # Signature
        self.add_signature_section()
        
        # QR Code
        qr_data = {
            "form_type": "vehicle_info",
            "form_id": self.form_id,
            "generated_at": datetime.now().isoformat()
        }
        self.add_qr_code(qr_data)
        
        return self.form_id
    
    def generate_driver_form(self) -> str:
        """Generate driver information form."""
        self.add_page()
        self.add_form_header(self.get_text("driver_form"))
        
        # Driver identification
        self.add_text_field(self.get_text("driver_name"), 120)
        self.add_text_field(self.get_text("license_type"), 80)
        self.add_text_field(self.get_text("license_expiry"), 80)
        self.add_number_field(self.get_text("experience_years"), 60)
        
        # Certifications
        cert_options = [
            self.get_text("adr_license"),
            self.get_text("forklift_license")
        ]
        self.add_checkbox_group(self.get_text("certifications"), cert_options)
        
        # Work preferences
        self.add_text_field(self.get_text("preferred_routes"), 150, False)
        self.add_text_field(self.get_text("availability"), 120)
        self.add_number_field(self.get_text("cost_per_hour"), 80)
        
        # Contact information
        self.add_text_field(self.get_text("phone_number"), 100)
        self.add_text_field(self.get_text("email"), 120)
        
        # Notes
        self.add_large_text_area(self.get_text("notes"))
        
        # Signature
        self.add_signature_section()
        
        # QR Code
        qr_data = {
            "form_type": "driver_info", 
            "form_id": self.form_id,
            "generated_at": datetime.now().isoformat()
        }
        self.add_qr_code(qr_data)
        
        return self.form_id
    
    def save_form(self, form_type: str) -> str:
        """Save the generated form to file."""
        os.makedirs(self.config.output_directory, exist_ok=True)
        
        filename = f"{form_type}_{self.form_id}.pdf"
        filepath = os.path.join(self.config.output_directory, filename)
        
        self.output(filepath)
        return filepath


# Convenience functions for easy form generation
def generate_delivery_form(config: FormConfig = None) -> str:
    """Generate a delivery order form."""
    generator = VRPFormGenerator(config)
    generator.generate_delivery_order_form()
    return generator.save_form("delivery_order")

def generate_vehicle_form(config: FormConfig = None) -> str:
    """Generate a vehicle information form."""
    generator = VRPFormGenerator(config)
    generator.generate_vehicle_form()
    return generator.save_form("vehicle_info")

def generate_driver_form(config: FormConfig = None) -> str:
    """Generate a driver information form."""
    generator = VRPFormGenerator(config)
    generator.generate_driver_form()
    return generator.save_form("driver_info")

def generate_all_forms(config: FormConfig = None) -> Dict[str, str]:
    """Generate all VRP forms."""
    paths = {}
    
    # Delivery form
    paths["delivery"] = generate_delivery_form(config)
    
    # Vehicle form
    paths["vehicle"] = generate_vehicle_form(config)
    
    # Driver form
    paths["driver"] = generate_driver_form(config)
    
    return paths


if __name__ == "__main__":
    # Demo: Generate all form types
    print("🚀 VRP Enhanced Form Generator Demo")
    print("=" * 50)
    
    # Configure for Italian company
    config = FormConfig(
        language="it",
        company_name="Optimize Logistics Solutions",
        output_directory="forms_output",
        include_qr_code=True
    )
    
    try:
        # Generate all forms
        form_paths = generate_all_forms(config)
        
        print("✅ Forms generated successfully:")
        for form_type, path in form_paths.items():
            print(f"   • {form_type.title()}: {path}")
        
        print(f"\n📂 All forms saved in: {config.output_directory}/")
        print("🔄 Forms include QR codes for digital processing")
        print("🌍 Multi-language support (IT/EN)")
        print("📋 Ready for database integration!")
        
    except Exception as e:
        print(f"❌ Error generating forms: {e}")
        import traceback
        traceback.print_exc()
