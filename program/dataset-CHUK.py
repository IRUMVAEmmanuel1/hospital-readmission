# chuk_healthcare_dataset_generator.py
import json
import random
import datetime
import uuid
import threading
import numpy as np
from faker import Faker

fake = Faker('en_US')  # English locale only

# --- Configuration ---
NUM_PATIENTS = 15000
NUM_DOCTORS = 200
NUM_NURSES = 500
NUM_DEPARTMENTS = 20
NUM_MEDICAL_EQUIPMENT = 300
NUM_MEDICATIONS = 800
NUM_APPOINTMENTS = 100000
NUM_ADMISSIONS = 25000
NUM_MEDICAL_RECORDS = 80000
NUM_LABORATORY_TESTS = 120000
NUM_PRESCRIPTIONS = 90000
NUM_BILLING_RECORDS = 85000
TIMESPAN_DAYS = 240  # January 2025 to August 2025
MAX_ITERATIONS = (NUM_APPOINTMENTS + NUM_ADMISSIONS + NUM_MEDICAL_RECORDS) * 2

# --- Initialization ---
np.random.seed(42)
random.seed(42)
Faker.seed(42)

print("Initializing CHUK Healthcare Dataset Generation...")

# --- ID Generators ---
def generate_patient_id():
    return f"CHUK_PAT_{uuid.uuid4().hex[:8].upper()}"

def generate_doctor_id():
    return f"CHUK_DOC_{uuid.uuid4().hex[:6].upper()}"

def generate_appointment_id():
    return f"CHUK_APT_{uuid.uuid4().hex[:10].upper()}"

def generate_admission_id():
    return f"CHUK_ADM_{uuid.uuid4().hex[:8].upper()}"

def generate_record_id():
    return f"CHUK_REC_{uuid.uuid4().hex[:10].upper()}"

def generate_test_id():
    return f"CHUK_LAB_{uuid.uuid4().hex[:8].upper()}"

def generate_prescription_id():
    return f"CHUK_PRX_{uuid.uuid4().hex[:8].upper()}"

def generate_bill_id():
    return f"CHUK_BIL_{uuid.uuid4().hex[:8].upper()}"

# --- Helper Functions ---
def get_rwandan_names():
    """Generate realistic Rwandan names"""
    first_names_male = ['Jean', 'Emmanuel', 'Patrick', 'Samuel', 'David', 'Christian', 'Joseph', 'Eric', 
                       'Claude', 'Vincent', 'Gilbert', 'Alain', 'Fabrice', 'Pierre', 'Paul', 'Antoine',
                       'Innocent', 'Faustin', 'Celestin', 'Bosco', 'Damascene', 'Fidele', 'Francois',
                       'Gratien', 'Janvier', 'Martin', 'Michel', 'Pascal', 'Robert', 'Sylvestre']
    
    first_names_female = ['Marie', 'Grace', 'Immaculee', 'Chantal', 'Angelique', 'Vestine', 'Josephine', 
                         'Solange', 'Yvonne', 'Esperance', 'Jeannette', 'Francine', 'Agnes', 'Beatrice',
                         'Christine', 'Delphine', 'Euphrasie', 'Gisele', 'Harriet', 'Ingabire', 'Jacqueline',
                         'Kevine', 'Louise', 'Marceline', 'Nadine', 'Odette', 'Petronille', 'Rosine',
                         'Sandrine', 'Therese', 'Valentine', 'Winnie', 'Xaverine', 'Yvette', 'Zainab']
    
    last_names = ['Uwimana', 'Mukamana', 'Niyonzima', 'Uwumuremyi', 'Hakizimana', 'Murekatete', 
                  'Nshimiyimana', 'Nyirahabimana', 'Bizimungu', 'Mukamusoni', 'Nsengimana', 
                  'Mukabatsinda', 'Habimana', 'Nyiramana', 'Rutayisire', 'Mukashema', 'Ntaganda',
                  'Uwizeye', 'Nsabimana', 'Munyangabe', 'Kayitesi', 'Byukusenge', 'Muhire', 'Ntawukuriryayo',
                  'Gasana', 'Kanyange', 'Munyakazi', 'Nzeyimana', 'Twagirumukiza', 'Uwimana', 'Nyampinga',
                  'Mutabazi', 'Niyigena', 'Harerimana', 'Murekezi', 'Kalisa', 'Sibomana', 'Tuyisenge']
    
    # Randomly choose gender-appropriate first name
    gender = random.choice(['Male', 'Female'])
    if gender == 'Male':
        first_name = random.choice(first_names_male)
    else:
        first_name = random.choice(first_names_female)
    
    last_name = random.choice(last_names)
    
    return first_name, last_name, gender

def get_medical_specialties():
    return ['Cardiology', 'Neurology', 'Oncology', 'Pediatrics', 'Obstetrics', 'Orthopedics', 
            'Dermatology', 'Emergency Medicine', 'Internal Medicine', 'Surgery', 'Radiology',
            'Psychiatry', 'Anesthesiology', 'Pathology', 'Family Medicine', 'Infectious Disease',
            'Gastroenterology', 'Endocrinology', 'Nephrology', 'Pulmonology']

def get_medical_conditions():
    return ['Hypertension', 'Diabetes Type 2', 'Malaria', 'Tuberculosis', 'HIV/AIDS', 
            'Pneumonia', 'Gastritis', 'Anemia', 'Arthritis', 'Asthma', 'Hepatitis B',
            'Typhoid', 'Urinary Tract Infection', 'Respiratory Infection', 'Skin Disease',
            'Heart Disease', 'Stroke', 'Cancer', 'Kidney Disease', 'Mental Health Disorder']

def get_lab_tests():
    return ['Complete Blood Count', 'Blood Glucose', 'Liver Function Test', 'Kidney Function Test',
            'Lipid Profile', 'Thyroid Function', 'HIV Test', 'Hepatitis Panel', 'Malaria Test',
            'Tuberculosis Test', 'Urinalysis', 'Stool Analysis', 'Chest X-Ray', 'ECG',
            'Ultrasound', 'CT Scan', 'MRI', 'Blood Culture', 'Pregnancy Test', 'PSA Test']

def get_medications():
    return ['Paracetamol', 'Amoxicillin', 'Ciprofloxacin', 'Metformin', 'Amlodipine', 
            'Atenolol', 'Omeprazole', 'Ibuprofen', 'Aspirin', 'Cotrimoxazole',
            'Artemether-Lumefantrine', 'Efavirenz', 'Tenofovir', 'Iron Tablets', 'Insulin',
            'Salbutamol', 'Prednisolone', 'Fluconazole', 'Metronidazole', 'Doxycycline']

# --- Department Generation ---
departments = []
specialties = get_medical_specialties()
for dept_id in range(NUM_DEPARTMENTS):
    department = {
        "department_id": f"DEPT_{dept_id:03d}",
        "name": specialties[dept_id] if dept_id < len(specialties) else f"Department {dept_id}",
        "head_doctor": None,  # Will be assigned later
        "location": f"Building {random.choice(['A', 'B', 'C', 'D'])}, Floor {random.randint(1, 5)}",
        "bed_capacity": random.randint(20, 100) if 'Emergency' not in specialties[dept_id] else 50,
        "equipment_count": random.randint(5, 25),
        "operational_hours": "24/7" if specialties[dept_id] in ['Emergency Medicine', 'Surgery'] else "08:00-17:00"
    }
    departments.append(department)

print(f"Generated {len(departments)} departments")

# --- Medical Staff Generation ---
doctors = []
nurses = []
specialties_list = get_medical_specialties()

# Generate Doctors
for doc_id in range(NUM_DOCTORS):
    first_name, last_name, gender = get_rwandan_names()
    specialty = random.choice(specialties_list)
    department = random.choice([d for d in departments if d["name"] == specialty or random.random() < 0.3])
    
    hire_date = fake.date_time_between(
        start_date=datetime.datetime(2020, 1, 1),
        end_date=datetime.datetime(2024, 12, 31)
    )
    
    doctor = {
        "doctor_id": generate_doctor_id(),
        "first_name": first_name,
        "last_name": last_name,
        "gender": gender,
        "specialty": specialty,
        "department_id": department["department_id"],
        "license_number": f"RW_MD_{random.randint(100000, 999999)}",
        "years_experience": random.randint(2, 30),
        "education": random.choice(["MD", "MD, PhD", "MD, MSc"]),
        "hire_date": hire_date.isoformat(),
        "contact_info": {
            "phone": f"+250{random.randint(700000000, 799999999)}",
            "email": f"{first_name.lower()}.{last_name.lower()}@chuk.rw"
        },
        "shift_pattern": random.choice(["Day", "Night", "Rotating"]),
        "consultation_fee": round(random.uniform(15000, 50000), 0)  # RWF
    }
    doctors.append(doctor)

# Assign head doctors to departments
for dept in departments:
    dept_doctors = [d for d in doctors if d["department_id"] == dept["department_id"]]
    if dept_doctors:
        dept["head_doctor"] = random.choice(dept_doctors)["doctor_id"]

# Generate Nurses
for nurse_id in range(NUM_NURSES):
    first_name, last_name, gender = get_rwandan_names()
    department = random.choice(departments)
    
    hire_date = fake.date_time_between(
        start_date=datetime.datetime(2018, 1, 1),
        end_date=datetime.datetime(2024, 12, 31)
    )
    
    nurse = {
        "nurse_id": f"CHUK_NUR_{uuid.uuid4().hex[:6].upper()}",
        "first_name": first_name,
        "last_name": last_name,
        "gender": gender,
        "department_id": department["department_id"],
        "license_number": f"RW_RN_{random.randint(100000, 999999)}",
        "education": random.choice(["Diploma in Nursing", "Bachelor in Nursing", "Advanced Diploma"]),
        "years_experience": random.randint(1, 25),
        "hire_date": hire_date.isoformat(),
        "shift_pattern": random.choice(["Day", "Night", "Rotating"]),
        "specialization": random.choice(["General", "ICU", "Pediatric", "Surgical", "Emergency"])
    }
    nurses.append(nurse)

print(f"Generated {len(doctors)} doctors and {len(nurses)} nurses")

# --- Patient Generation ---
patients = []
for patient_id in range(NUM_PATIENTS):
    first_name, last_name, gender = get_rwandan_names()
    birth_date = fake.date_time_between(
        start_date=datetime.datetime(1940, 1, 1),
        end_date=datetime.datetime(2020, 12, 31)
    )
    
    registration_date = fake.date_time_between(
        start_date=datetime.datetime(2024, 1, 1),
        end_date=datetime.datetime(2025, 1, 31)
    )
    
    # Calculate age
    today = datetime.datetime.now()
    age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
    
    patient = {
        "patient_id": generate_patient_id(),
        "first_name": first_name,
        "last_name": last_name,
        "date_of_birth": birth_date.date().isoformat(),
        "age": age,
        "gender": random.choice(["Male", "Female"]),
        "blood_type": random.choice(["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]),
        "contact_info": {
            "phone": f"+250{random.randint(700000000, 799999999)}",
            "email": f"{first_name.lower()}.{last_name.lower()}@gmail.com" if random.random() < 0.6 else None,
            "address": {
                "district": random.choice(["Nyarugenge", "Gasabo", "Kicukiro"]),
                "sector": fake.city(),
                "cell": fake.street_name(),
                "village": fake.street_name()
            }
        },
        "insurance_info": {
            "type": random.choice(["Mutuelle de Sante", "RAMA", "MMI", "Private", "None"]),
            "policy_number": f"INS_{random.randint(100000, 999999)}" if random.random() < 0.85 else None,
            "coverage_percentage": random.choice([0, 80, 90, 100]) if random.random() < 0.85 else 0
        },
        "emergency_contact": {
            "name": fake.name(),
            "relationship": random.choice(["Spouse", "Parent", "Sibling", "Child", "Friend"]),
            "phone": f"+250{random.randint(700000000, 799999999)}"
        },
        "registration_date": registration_date.isoformat(),
        "medical_history": random.sample(get_medical_conditions(), k=random.randint(0, 3)),
        "allergies": random.sample(["Penicillin", "Aspirin", "Latex", "Nuts", "Shellfish"], k=random.randint(0, 2))
    }
    patients.append(patient)

print(f"Generated {len(patients)} patients")

# --- Medical Equipment Generation ---
equipment_types = ["X-Ray Machine", "CT Scanner", "MRI Machine", "Ultrasound", "ECG Machine", 
                  "Ventilator", "Defibrillator", "Patient Monitor", "Surgical Robot", "Dialysis Machine",
                  "Anesthesia Machine", "Blood Analyzer", "Microscope", "Centrifuge", "Autoclave"]

medical_equipment = []
for eq_id in range(NUM_MEDICAL_EQUIPMENT):
    purchase_date = fake.date_time_between(
        start_date=datetime.datetime(2015, 1, 1),
        end_date=datetime.datetime(2024, 12, 31)
    )
    
    equipment = {
        "equipment_id": f"CHUK_EQ_{eq_id:04d}",
        "name": random.choice(equipment_types),
        "manufacturer": random.choice(["Siemens", "GE Healthcare", "Philips", "Canon", "Mindray"]),
        "model": f"Model_{random.randint(1000, 9999)}",
        "serial_number": f"SN_{uuid.uuid4().hex[:8].upper()}",
        "department_id": random.choice(departments)["department_id"],
        "purchase_date": purchase_date.isoformat(),
        "warranty_expiry": (purchase_date + datetime.timedelta(days=random.randint(365, 1825))).isoformat(),
        "status": random.choice(["Operational", "Under Maintenance", "Out of Service"]),
        "last_maintenance": fake.date_time_between(
            start_date=purchase_date,
            end_date=datetime.datetime.now()
        ).isoformat(),
        "usage_hours": random.randint(1000, 50000),
        "cost": round(random.uniform(50000, 5000000), 0)  # RWF
    }
    medical_equipment.append(equipment)

print(f"Generated {len(medical_equipment)} medical equipment items")

# --- Generate Time-Series Data ---
appointments = []
admissions = []
medical_records = []
laboratory_tests = []
prescriptions = []
billing_records = []

print("Generating time-series medical data...")

start_date = datetime.datetime(2025, 1, 1)
end_date = datetime.datetime.now()

# Generate Appointments
for apt_id in range(NUM_APPOINTMENTS):
    patient = random.choice(patients)
    doctor = random.choice(doctors)
    appointment_date = fake.date_time_between(start_date=start_date, end_date=end_date)
    
    appointment = {
        "appointment_id": generate_appointment_id(),
        "patient_id": patient["patient_id"],
        "doctor_id": doctor["doctor_id"],
        "department_id": doctor["department_id"],
        "appointment_date": appointment_date.isoformat(),
        "appointment_time": f"{random.randint(8, 17):02d}:{random.choice(['00', '30'])}",
        "type": random.choice(["Consultation", "Follow-up", "Emergency", "Surgery", "Routine Check"]),
        "status": random.choice(["Scheduled", "Completed", "Cancelled", "No-Show"]),
        "chief_complaint": random.choice(["Chest pain", "Headache", "Fever", "Cough", "Abdominal pain", 
                                        "Back pain", "Fatigue", "Shortness of breath"]),
        "priority": random.choice(["Low", "Medium", "High", "Critical"]),
        "estimated_duration": random.randint(15, 120),  # minutes
        "notes": "Patient appointment for " + random.choice(get_medical_conditions()).lower()
    }
    appointments.append(appointment)

# Generate Admissions
for adm_id in range(NUM_ADMISSIONS):
    patient = random.choice(patients)
    doctor = random.choice(doctors)
    admission_date = fake.date_time_between(start_date=start_date, end_date=end_date)
    
    # Calculate discharge date (some still admitted)
    discharge_date = None
    if random.random() < 0.7:  # 70% discharged
        discharge_date = admission_date + datetime.timedelta(days=random.randint(1, 30))
        if discharge_date > datetime.datetime.now():
            discharge_date = None
    
    admission = {
        "admission_id": generate_admission_id(),
        "patient_id": patient["patient_id"],
        "admitting_doctor_id": doctor["doctor_id"],
        "department_id": doctor["department_id"],
        "admission_date": admission_date.isoformat(),
        "discharge_date": discharge_date.isoformat() if discharge_date else None,
        "admission_type": random.choice(["Emergency", "Elective", "Transfer", "Observation"]),
        "room_number": f"{random.randint(1, 5)}{random.randint(10, 99)}",
        "bed_number": random.randint(1, 4),
        "primary_diagnosis": random.choice(get_medical_conditions()),
        "secondary_diagnoses": random.sample(get_medical_conditions(), k=random.randint(0, 2)),
        "admission_reason": "Admitted for treatment of " + random.choice(get_medical_conditions()).lower(),
        "discharge_reason": random.choice(["Improved", "Transferred", "Against Medical Advice", "Deceased"]) if discharge_date else None,
        "total_cost": round(random.uniform(50000, 2000000), 0) if discharge_date else None  # RWF
    }
    admissions.append(admission)

# Generate Medical Records
for rec_id in range(NUM_MEDICAL_RECORDS):
    patient = random.choice(patients)
    doctor = random.choice(doctors)
    record_date = fake.date_time_between(start_date=start_date, end_date=end_date)
    
    # Vital signs
    vital_signs = {
        "blood_pressure": f"{random.randint(90, 180)}/{random.randint(60, 120)}",
        "heart_rate": random.randint(60, 120),
        "temperature": round(random.uniform(36.0, 39.5), 1),
        "respiratory_rate": random.randint(12, 25),
        "oxygen_saturation": random.randint(85, 100),
        "weight": round(random.uniform(40, 120), 1),
        "height": random.randint(140, 200)
    }
    
    medical_record = {
        "record_id": generate_record_id(),
        "patient_id": patient["patient_id"],
        "doctor_id": doctor["doctor_id"],
        "visit_date": record_date.isoformat(),
        "visit_type": random.choice(["Consultation", "Follow-up", "Emergency", "Routine"]),
        "chief_complaint": random.choice(["Fever", "Pain", "Cough", "Fatigue", "Nausea"]),
        "history_of_present_illness": "Patient presents with symptoms of " + random.choice(get_medical_conditions()).lower(),
        "physical_examination": "Physical exam reveals findings consistent with primary complaint",
        "vital_signs": vital_signs,
        "diagnosis": random.choice(get_medical_conditions()),
        "treatment_plan": "Treatment plan developed based on diagnosis and patient condition",
        "medications_prescribed": random.sample(get_medications(), k=random.randint(1, 4)),
        "follow_up_required": random.choice([True, False]),
        "follow_up_date": (record_date + datetime.timedelta(days=random.randint(7, 30))).isoformat() if random.choice([True, False]) else None
    }
    medical_records.append(medical_record)

# Generate Laboratory Tests
lab_tests_list = get_lab_tests()
for test_id in range(NUM_LABORATORY_TESTS):
    patient = random.choice(patients)
    test_date = fake.date_time_between(start_date=start_date, end_date=end_date)
    test_name = random.choice(lab_tests_list)
    
    # Generate realistic test results based on test type
    test_results = {}
    if "Blood Count" in test_name:
        test_results = {
            "hemoglobin": f"{round(random.uniform(10, 18), 1)} g/dL",
            "white_blood_cells": f"{random.randint(4000, 12000)} cells/µL",
            "platelets": f"{random.randint(150000, 400000)} cells/µL"
        }
    elif "Glucose" in test_name:
        test_results = {"glucose_level": f"{round(random.uniform(70, 200), 1)} mg/dL"}
    elif "HIV" in test_name:
        test_results = {"result": random.choice(["Negative", "Positive", "Indeterminate"])}
    else:
        test_results = {"result": random.choice(["Normal", "Abnormal", "Borderline"])}
    
    laboratory_test = {
        "test_id": generate_test_id(),
        "patient_id": patient["patient_id"],
        "test_name": test_name,
        "test_date": test_date.isoformat(),
        "ordered_by": random.choice(doctors)["doctor_id"],
        "sample_collected_date": test_date.isoformat(),
        "result_date": (test_date + datetime.timedelta(hours=random.randint(2, 72))).isoformat(),
        "test_results": test_results,
        "reference_range": "Within normal limits" if test_results.get("result") == "Normal" else "See detailed report",
        "status": random.choice(["Completed", "Pending", "In Progress"]),
        "cost": round(random.uniform(5000, 50000), 0),  # RWF
        "lab_technician": fake.name()
    }
    laboratory_tests.append(laboratory_test)

# Generate Prescriptions
medications_list = get_medications()
for prx_id in range(NUM_PRESCRIPTIONS):
    patient = random.choice(patients)
    doctor = random.choice(doctors)
    prescription_date = fake.date_time_between(start_date=start_date, end_date=end_date)
    
    # Generate multiple medications per prescription
    prescription_items = []
    for _ in range(random.randint(1, 4)):
        medication = random.choice(medications_list)
        prescription_items.append({
            "medication_name": medication,
            "dosage": random.choice(["250mg", "500mg", "1g", "2.5mg", "5mg", "10mg"]),
            "frequency": random.choice(["Once daily", "Twice daily", "Three times daily", "Four times daily", "As needed"]),
            "duration": f"{random.randint(3, 30)} days",
            "quantity": random.randint(10, 90),
            "unit_cost": round(random.uniform(500, 10000), 0),  # RWF
            "total_cost": 0  # Will be calculated
        })
        prescription_items[-1]["total_cost"] = prescription_items[-1]["quantity"] * prescription_items[-1]["unit_cost"]
    
    prescription = {
        "prescription_id": generate_prescription_id(),
        "patient_id": patient["patient_id"],
        "doctor_id": doctor["doctor_id"],
        "prescription_date": prescription_date.isoformat(),
        "medications": prescription_items,
        "total_cost": sum(item["total_cost"] for item in prescription_items),
        "status": random.choice(["Active", "Completed", "Discontinued"]),
        "pharmacy_notes": "Dispensed as prescribed" if random.random() < 0.8 else "Partial dispensing due to stock",
        "refills_remaining": random.randint(0, 3)
    }
    prescriptions.append(prescription)

# Generate Billing Records
for bill_id in range(NUM_BILLING_RECORDS):
    patient = random.choice(patients)
    billing_date = fake.date_time_between(start_date=start_date, end_date=end_date)
    
    # Generate multiple services per bill
    services = []
    service_types = ["Consultation", "Laboratory Test", "Medication", "Procedure", "Admission", "Surgery"]
    
    for _ in range(random.randint(1, 5)):
        service_type = random.choice(service_types)
        cost = round(random.uniform(5000, 200000), 0)  # RWF
        services.append({
            "service_type": service_type,
            "description": f"{service_type} - {fake.sentence(nb_words=4)}",
            "cost": cost,
            "date": billing_date.isoformat()
        })
    
    subtotal = sum(service["cost"] for service in services)
    insurance_coverage = patient["insurance_info"]["coverage_percentage"]
    insurance_amount = round(subtotal * (insurance_coverage / 100), 0) if insurance_coverage > 0 else 0
    patient_amount = subtotal - insurance_amount
    
    billing_record = {
        "billing_id": generate_bill_id(),
        "patient_id": patient["patient_id"],
        "billing_date": billing_date.isoformat(),
        "services": services,
        "subtotal": subtotal,
        "insurance_coverage_percent": insurance_coverage,
        "insurance_amount": insurance_amount,
        "patient_amount": patient_amount,
        "total_amount": subtotal,
        "payment_status": random.choice(["Paid", "Pending", "Partial", "Overdue"]),
        "payment_method": random.choice(["Cash", "Mobile Money", "Bank Transfer", "Insurance", "Credit"]),
        "invoice_number": f"INV_{random.randint(100000, 999999)}",
        "due_date": (billing_date + datetime.timedelta(days=30)).isoformat()
    }
    billing_records.append(billing_record)

print(f"""
Generated time-series data:
- Appointments: {len(appointments):,}
- Admissions: {len(admissions):,}
- Medical Records: {len(medical_records):,}
- Laboratory Tests: {len(laboratory_tests):,}
- Prescriptions: {len(prescriptions):,}
- Billing Records: {len(billing_records):,}
""")

# --- Data Export ---
def json_serializer(obj):
    """Custom JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")

print("Saving CHUK Healthcare datasets...")

# Save all datasets
datasets = {
    "departments": departments,
    "doctors": doctors,
    "nurses": nurses,
    "patients": patients,
    "medical_equipment": medical_equipment,
    "appointments": appointments,
    "admissions": admissions,
    "medical_records": medical_records,
    "laboratory_tests": laboratory_tests,
    "prescriptions": prescriptions,
    "billing_records": billing_records
}

for name, data in datasets.items():
    # Save smaller datasets as single files
    if len(data) <= 50000:
        with open(f"chuk_{name}.json", "w") as f:
            json.dump(data, f, default=json_serializer, indent=2)
    else:
        # Save larger datasets in chunks
        CHUNK_SIZE = 50000
        for i in range(0, len(data), CHUNK_SIZE):
            chunk = data[i:i+CHUNK_SIZE]
            with open(f"chuk_{name}_{i//CHUNK_SIZE}.json", "w") as f:
                json.dump(chunk, f, default=json_serializer, indent=2)

# Generate summary statistics
summary = {
    "dataset_info": {
        "hospital_name": "Centre Hospitalier Universitaire de Kigali (CHUK)",
        "data_period": f"{start_date.date()} to {end_date.date()}",
        "generation_date": datetime.datetime.now().isoformat(),
        "total_records": sum(len(data) for data in datasets.values())
    },
    "entity_counts": {name: len(data) for name, data in datasets.items()},
    "data_quality_notes": [
        "All patient data is synthetic and complies with privacy regulations",
        "Medical conditions and treatments are realistic but randomly assigned",
        "Cost figures are in Rwandan Francs (RWF)",
        "Time-series data spans from January 2025 to present",
        "Insurance coverage reflects typical Rwandan health insurance patterns"
    ]
}

with open("chuk_dataset_summary.json", "w") as f:
    json.dump(summary, f, default=json_serializer, indent=2)

print(f"""
🏥 CHUK Healthcare Dataset Generation Complete! 🏥

Dataset Summary:
- Departments: {len(departments):,}
- Medical Staff: {len(doctors) + len(nurses):,} (Doctors: {len(doctors)}, Nurses: {len(nurses)})
- Patients: {len(patients):,}
- Medical Equipment: {len(medical_equipment):,}
- Appointments: {len(appointments):,}
- Admissions: {len(admissions):,}
- Medical Records: {len(medical_records):,}
- Laboratory Tests: {len(laboratory_tests):,}
- Prescriptions: {len(prescriptions):,}
- Billing Records: {len(billing_records):,}

Total Records: {sum(len(data) for data in datasets.values()):,}

Files generated:
- chuk_laboratory_tests.json (or chunked files if large)
- chuk_prescriptions.json (or chunked files if large)
- chuk_billing_records.json (or chunked files if large)
- chuk_dataset_summary.json

🔍 Key Features for Big Data Analytics:

MongoDB Document Model Advantages:
✓ Patient profiles with embedded medical history, insurance info, and contact details
✓ Complex prescription documents with multiple medications and dosages
✓ Hierarchical department structure with embedded staff assignments
✓ Rich medical records with embedded vital signs and examination details

HBase Wide-Column Model Advantages:
✓ Time-series appointment data (patient_id + timestamp for efficient querying)
✓ Laboratory test results over time (test_id + date for trend analysis)
✓ Patient admission/discharge patterns (admission_id + date ranges)
✓ Equipment usage tracking (equipment_id + maintenance_date)

Apache Spark Processing Opportunities:
✓ Patient cohort analysis across multiple data sources
✓ Medical equipment utilization and maintenance prediction
✓ Revenue analytics combining billing, insurance, and treatment data
✓ Disease outbreak detection through appointment and diagnosis patterns
✓ Resource optimization (doctor scheduling, bed utilization, equipment allocation)

Integration Analytics Examples:
✓ Patient Journey Analysis: Track complete patient flow from appointment → admission → treatment → billing
✓ Clinical Decision Support: Combine patient history + lab results + medication interactions
✓ Hospital Operations Dashboard: Real-time bed occupancy + staff scheduling + equipment status
✓ Financial Analytics: Insurance reimbursement patterns + patient payment analysis
✓ Quality Metrics: Treatment outcomes + readmission rates + patient satisfaction

Data Relationships for Complex Queries:
• Patient → Appointments → Medical Records → Lab Tests → Prescriptions → Billing
• Doctor → Department → Equipment → Patients treated
• Insurance Type → Coverage Patterns → Payment Methods → Revenue Analysis
• Time-based: Daily admissions, Monthly revenue, Seasonal disease patterns

Perfect for your thesis requirements:
✅ Multi-model database design (MongoDB + HBase)
✅ Large-scale data processing (Apache Spark)
✅ Real healthcare domain with realistic relationships
✅ Time-series data from January 2025 onwards
✅ Complex analytics opportunities across all three technologies
✅ Scalable data volumes suitable for big data demonstrations

🚀 Ready to implement your multi-faceted analytics system!

To run this generator:
1. Install dependencies: pip install faker numpy
2. Run: python chuk_healthcare_dataset_generator.py
3. Wait for generation to complete (may take several minutes)
4. Use generated JSON files for your MongoDB, HBase, and Spark implementations
""")
