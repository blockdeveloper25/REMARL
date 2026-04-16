"""
remarl/sim/scenario_gen.py
--------------------------
Synthetic Requirements Engineering scenario generator.

Each scenario is a complete "project brief" with:
  - rough_idea       : the vague input a stakeholder would give
  - domain           : the application domain
  - ground_truth_reqs: the correct functional requirements (oracle answer)
  - hidden_reqs      : subset the agents must DISCOVER through elicitation
  - nfr              : non-functional requirements
  - stakeholders     : list of stakeholder personas with interests
  - domain_entities  : key entities the modeler should extract
  - conflicts        : intentional contradictions to test the negotiator

Design principle:
  hidden_reqs are randomly withheld from the initial prompt.
  If the agents elicit well, they will surface them.
  The Oracle scores coverage against ground_truth_reqs.
  This gives a clean training signal without needing real projects.
"""

import json
import random
import pathlib
import hashlib
from dataclasses import dataclass, field, asdict
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  Data classes
# ─────────────────────────────────────────────

@dataclass
class Stakeholder:
    name: str
    role: str
    primary_interest: str
    secondary_interest: str
    conflict_with: Optional[str] = None  # role this stakeholder conflicts with


@dataclass
class Scenario:
    scenario_id: str
    domain: str
    rough_idea: str
    ground_truth_reqs: List[str]
    hidden_reqs: List[str]          # subset of ground_truth that starts hidden
    visible_reqs: List[str]         # ground_truth minus hidden (shown in prompt)
    nfr: List[str]
    stakeholders: List[Stakeholder]
    domain_entities: List[str]
    conflicts: List[dict]           # [{"req_a": ..., "req_b": ..., "type": ...}]
    difficulty: str                 # "easy" | "medium" | "hard"

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Scenario":
        d["stakeholders"] = [Stakeholder(**s) for s in d["stakeholders"]]
        return cls(**d)


# ─────────────────────────────────────────────
#  Domain template library
#  30 domains across 6 sectors
# ─────────────────────────────────────────────

DOMAIN_TEMPLATES = [

    # ── SECTOR 1: E-COMMERCE & RETAIL ────────────────────────────────────

    {
        "domain": "e_commerce_marketplace",
        "rough_idea": "An online marketplace where sellers can list products and buyers can purchase them securely.",
        "ground_truth_reqs": [
            "The system shall allow sellers to register, create a store profile, and list products with images, descriptions, and prices.",
            "The system shall allow buyers to search for products by keyword, category, and price range.",
            "The system shall provide a shopping cart that persists across user sessions.",
            "The system shall process payments via credit card, debit card, and PayPal.",
            "The system shall send order confirmation and shipping update emails to buyers.",
            "The system shall allow buyers to leave ratings and reviews for completed purchases.",
            "The system shall provide sellers with a dashboard showing sales, revenue, and inventory.",
            "The system shall enforce a return and refund workflow with seller approval.",
        ],
        "nfr": [
            "Page load time shall not exceed 2 seconds under normal load.",
            "The system shall be available 99.9% of the time.",
            "All payment data shall be encrypted using AES-256.",
            "The system shall comply with PCI-DSS standards.",
        ],
        "stakeholders": [
            Stakeholder("Alice", "buyer", "low prices and fast delivery", "easy returns", "seller"),
            Stakeholder("Bob", "seller", "high visibility and low commission fees", "analytics", "buyer"),
            Stakeholder("Carol", "platform_admin", "fraud prevention and compliance", "revenue", None),
        ],
        "domain_entities": ["User", "Seller", "Product", "Cart", "Order", "Payment", "Review", "Category"],
        "conflicts": [
            {"req_a": "Buyers shall receive full refunds within 24 hours.",
             "req_b": "Sellers shall have 7 days to approve or deny refund requests.",
             "type": "temporal_conflict"},
        ],
        "difficulty": "medium",
    },

    {
        "domain": "subscription_box",
        "rough_idea": "A service that sends personalised monthly product boxes to subscribers based on their preferences.",
        "ground_truth_reqs": [
            "The system shall allow users to complete a preference questionnaire during onboarding.",
            "The system shall generate a personalised box selection based on user preferences and past ratings.",
            "The system shall support monthly, quarterly, and annual subscription plans.",
            "The system shall allow users to pause or cancel their subscription at any time.",
            "The system shall send a box preview notification 5 days before shipping.",
            "The system shall allow users to swap up to 2 products from their upcoming box.",
            "The system shall charge subscriptions automatically on the renewal date.",
            "The system shall track delivery status and notify users at each shipping milestone.",
        ],
        "nfr": [
            "Personalisation algorithm shall run within 500ms.",
            "System shall handle 10,000 concurrent active subscribers.",
            "GDPR compliance for user preference data.",
        ],
        "stakeholders": [
            Stakeholder("Dana", "subscriber", "personalised relevant products", "surprise factor", None),
            Stakeholder("Eve", "curator", "manageable curation workload", "product diversity", None),
            Stakeholder("Frank", "logistics_manager", "predictable fulfilment volumes", "on-time delivery", None),
        ],
        "domain_entities": ["Subscriber", "Box", "Product", "Subscription", "Preference", "Shipment"],
        "conflicts": [
            {"req_a": "Users shall be able to cancel with no notice period.",
             "req_b": "Boxes shall be prepared 10 days before the shipping date, making late cancellations non-refundable.",
             "type": "business_rule_conflict"},
        ],
        "difficulty": "medium",
    },

    # ── SECTOR 2: HEALTHCARE ─────────────────────────────────────────────

    {
        "domain": "patient_portal",
        "rough_idea": "A web portal where patients can view their medical records, book appointments, and message their doctors.",
        "ground_truth_reqs": [
            "The system shall allow patients to register using their NHS/insurance number and verify their identity.",
            "The system shall display a patient's medical history, test results, and current prescriptions.",
            "The system shall allow patients to book, reschedule, and cancel appointments online.",
            "The system shall provide a secure messaging channel between patients and their assigned GP.",
            "The system shall allow doctors to update patient records and issue electronic prescriptions.",
            "The system shall send appointment reminders via SMS and email 24 hours before.",
            "The system shall allow patients to download their records as a PDF.",
            "The system shall enforce role-based access so patients cannot view other patients' records.",
        ],
        "nfr": [
            "System shall comply with HL7 FHIR standards for health data.",
            "All data at rest shall be encrypted using AES-256.",
            "System shall comply with HIPAA and GDPR.",
            "Authentication shall use two-factor authentication.",
            "System uptime shall be 99.95%.",
        ],
        "stakeholders": [
            Stakeholder("Grace", "patient", "easy access to own records", "privacy", "doctor"),
            Stakeholder("Henry", "gp_doctor", "efficient appointment management", "clinical accuracy", "patient"),
            Stakeholder("Irene", "hospital_admin", "regulatory compliance", "cost reduction", None),
        ],
        "domain_entities": ["Patient", "Doctor", "Appointment", "MedicalRecord", "Prescription", "Message", "TestResult"],
        "conflicts": [
            {"req_a": "Patients shall have immediate access to all test results.",
             "req_b": "Doctors shall review and annotate test results before patient release to prevent misinterpretation.",
             "type": "access_timing_conflict"},
        ],
        "difficulty": "hard",
    },

    {
        "domain": "mental_health_app",
        "rough_idea": "A mobile app for daily mood tracking, guided meditation, and connecting users with therapists.",
        "ground_truth_reqs": [
            "The system shall allow users to log their mood on a 1-10 scale with optional notes each day.",
            "The system shall display a mood trend chart over the past 30 days.",
            "The system shall provide a library of guided meditation sessions categorised by duration and goal.",
            "The system shall match users with licensed therapists based on their stated concerns.",
            "The system shall support video, voice, and text therapy sessions within the app.",
            "The system shall send a daily check-in notification at a user-configured time.",
            "The system shall detect 3 consecutive low-mood entries and prompt access to crisis resources.",
            "The system shall allow users to export their mood data for sharing with their therapist.",
        ],
        "nfr": [
            "All therapy session data shall be end-to-end encrypted.",
            "Crisis detection algorithm shall have false negative rate below 5%.",
            "App shall function with degraded connectivity for mood logging.",
            "HIPAA compliance mandatory.",
        ],
        "stakeholders": [
            Stakeholder("James", "end_user", "private mood tracking without stigma", "therapist access", None),
            Stakeholder("Karen", "therapist", "structured patient data before sessions", "session scheduling", None),
            Stakeholder("Leo", "clinical_director", "evidence-based intervention triggers", "liability management", None),
        ],
        "domain_entities": ["User", "MoodEntry", "Therapist", "Session", "MeditationTrack", "CrisisAlert"],
        "conflicts": [
            {"req_a": "User mood data shall never be shared without explicit consent.",
             "req_b": "In the event of crisis indicators, the system shall notify an emergency contact automatically.",
             "type": "privacy_safety_conflict"},
        ],
        "difficulty": "hard",
    },

    # ── SECTOR 3: EDUCATION ──────────────────────────────────────────────

    {
        "domain": "online_learning_platform",
        "rough_idea": "An e-learning platform where instructors create courses and students learn at their own pace.",
        "ground_truth_reqs": [
            "The system shall allow instructors to create courses with video lectures, quizzes, and assignments.",
            "The system shall allow students to enrol in free and paid courses.",
            "The system shall track student progress and resume from the last watched position.",
            "The system shall generate a certificate of completion when a student passes all assessments.",
            "The system shall provide a discussion forum per course for student-instructor interaction.",
            "The system shall support multiple-choice, short answer, and coding exercise question types.",
            "The system shall allow instructors to set course prerequisites.",
            "The system shall provide instructors with analytics on student engagement and completion rates.",
        ],
        "nfr": [
            "Video streaming shall adapt to available bandwidth automatically.",
            "Platform shall support 50,000 concurrent learners.",
            "WCAG 2.1 AA accessibility compliance.",
            "System shall support English, Spanish, French, and Mandarin.",
        ],
        "stakeholders": [
            Stakeholder("Mia", "student", "self-paced affordable learning", "recognised certificates", None),
            Stakeholder("Noah", "instructor", "easy course creation tools", "revenue share", "student"),
            Stakeholder("Olivia", "platform_owner", "content quality and platform growth", "monetisation", None),
        ],
        "domain_entities": ["Student", "Instructor", "Course", "Lecture", "Quiz", "Certificate", "Forum", "Enrolment"],
        "conflicts": [
            {"req_a": "Instructors shall set their own pricing with no restrictions.",
             "req_b": "Platform shall enforce a maximum course price of £200 to maintain accessibility.",
             "type": "pricing_conflict"},
        ],
        "difficulty": "medium",
    },

    {
        "domain": "school_management_system",
        "rough_idea": "A system for schools to manage student records, timetables, attendance, and parent communication.",
        "ground_truth_reqs": [
            "The system shall maintain a complete academic record for each student including grades and attendance.",
            "The system shall generate class timetables based on teacher availability and room allocation.",
            "The system shall record daily attendance per class and flag absences to form teachers.",
            "The system shall allow teachers to submit grades for assignments and exams.",
            "The system shall generate progress reports each term and make them available to parents.",
            "The system shall provide a messaging system between teachers and parents.",
            "The system shall allow parents to submit absence notifications.",
            "The system shall track and report on SEND (special educational needs) student support plans.",
        ],
        "nfr": [
            "All student data shall comply with FERPA and GDPR.",
            "System shall be accessible on tablets used in classrooms.",
            "System shall support batch import of student data via CSV.",
        ],
        "stakeholders": [
            Stakeholder("Paul", "teacher", "minimal admin overhead", "clear grade tracking", None),
            Stakeholder("Quinn", "parent", "visibility of child's progress", "direct teacher communication", None),
            Stakeholder("Rachel", "headteacher", "whole-school analytics", "ofsted compliance", None),
        ],
        "domain_entities": ["Student", "Teacher", "Parent", "Class", "Timetable", "Attendance", "Grade", "Report"],
        "conflicts": [
            {"req_a": "Parents shall have real-time access to their child's grades.",
             "req_b": "Teachers shall have the ability to lock grades during moderation periods.",
             "type": "access_conflict"},
        ],
        "difficulty": "medium",
    },

    # ── SECTOR 4: FINTECH & BANKING ──────────────────────────────────────

    {
        "domain": "personal_finance_app",
        "rough_idea": "A mobile app that connects to bank accounts and helps users track spending, set budgets, and save money.",
        "ground_truth_reqs": [
            "The system shall connect to user bank accounts via Open Banking API (PSD2 compliant).",
            "The system shall automatically categorise transactions into spending categories.",
            "The system shall allow users to set monthly budgets per category.",
            "The system shall alert users when they reach 80% of a budget limit.",
            "The system shall display net worth by aggregating all connected accounts.",
            "The system shall allow users to set savings goals with target amounts and dates.",
            "The system shall generate a monthly spending report with category breakdown.",
            "The system shall allow manual transaction entry for cash spending.",
        ],
        "nfr": [
            "Bank connection shall use read-only OAuth tokens — no write access.",
            "All financial data shall be encrypted at rest and in transit.",
            "FCA regulated and PSD2 compliant.",
            "App shall not store full bank credentials.",
        ],
        "stakeholders": [
            Stakeholder("Sam", "end_user", "clear spending insight", "financial privacy", None),
            Stakeholder("Tara", "product_manager", "engagement and retention", "monetisation via premium tier", None),
            Stakeholder("Uma", "compliance_officer", "FCA and PSD2 compliance", "data minimisation", None),
        ],
        "domain_entities": ["User", "BankAccount", "Transaction", "Budget", "Category", "SavingsGoal", "Report"],
        "conflicts": [
            {"req_a": "System shall retain transaction history indefinitely for trend analysis.",
             "req_b": "GDPR requires data to be deleted upon user request within 30 days.",
             "type": "retention_conflict"},
        ],
        "difficulty": "hard",
    },

    {
        "domain": "peer_lending_platform",
        "rough_idea": "A platform where individuals can lend money to small businesses and earn interest returns.",
        "ground_truth_reqs": [
            "The system shall allow borrowers to submit loan applications with business details and loan purpose.",
            "The system shall perform automated credit risk assessment on borrower applications.",
            "The system shall allow lenders to browse verified loan listings and commit funds.",
            "The system shall distribute borrower repayments proportionally to all contributing lenders.",
            "The system shall provide a secondary market where lenders can sell their loan parts.",
            "The system shall send monthly statements to lenders showing interest earned.",
            "The system shall enforce FCA-regulated investor limits for retail investors.",
            "The system shall place defaulted loans into a collections workflow automatically.",
        ],
        "nfr": [
            "Platform shall be FCA authorised and regulated.",
            "Credit scoring model shall be explainable (no black-box decisions).",
            "All fund transfers shall use ring-fenced client money accounts.",
            "System shall pass annual penetration tests.",
        ],
        "stakeholders": [
            Stakeholder("Victor", "lender", "attractive returns with managed risk", "liquidity via secondary market", "borrower"),
            Stakeholder("Wendy", "borrower", "fast approval and fair rates", "privacy of financials", "lender"),
            Stakeholder("Xavier", "risk_officer", "default rate below 3%", "regulatory compliance", None),
        ],
        "domain_entities": ["Lender", "Borrower", "LoanApplication", "LoanListing", "Investment", "Repayment", "SecondaryMarket"],
        "conflicts": [
            {"req_a": "Borrowers shall receive loan decisions within 24 hours.",
             "req_b": "Risk assessment shall include manual review for loans above £50,000.",
             "type": "speed_thoroughness_conflict"},
        ],
        "difficulty": "hard",
    },

    # ── SECTOR 5: LOGISTICS & IoT ────────────────────────────────────────

    {
        "domain": "fleet_management",
        "rough_idea": "A system to track a company's delivery fleet in real time, optimise routes, and manage vehicle maintenance.",
        "ground_truth_reqs": [
            "The system shall display real-time GPS location of all vehicles on a map.",
            "The system shall calculate and suggest optimal delivery routes based on traffic data.",
            "The system shall send alerts when a vehicle deviates from its planned route.",
            "The system shall track fuel consumption per vehicle and flag inefficient driving behaviour.",
            "The system shall schedule preventive maintenance based on mileage and engine hours.",
            "The system shall allow drivers to log delivery confirmations via mobile app.",
            "The system shall generate end-of-day reports on deliveries completed, failed, and pending.",
            "The system shall integrate with the company's existing ERP for order data.",
        ],
        "nfr": [
            "GPS tracking shall update every 30 seconds.",
            "Mobile app shall work offline and sync when connectivity is restored.",
            "System shall handle a fleet of up to 500 vehicles.",
            "Driver data shall comply with GDPR.",
        ],
        "stakeholders": [
            Stakeholder("Yara", "fleet_manager", "full vehicle visibility", "cost reduction", None),
            Stakeholder("Zack", "driver", "simple mobile interface", "privacy of location outside working hours", "fleet_manager"),
            Stakeholder("Anna", "operations_director", "delivery SLA compliance", "ERP integration", None),
        ],
        "domain_entities": ["Vehicle", "Driver", "Route", "Delivery", "MaintenanceSchedule", "FuelLog", "Alert"],
        "conflicts": [
            {"req_a": "System shall track driver location continuously during working hours.",
             "req_b": "Drivers shall have the ability to disable location tracking during breaks.",
             "type": "surveillance_privacy_conflict"},
        ],
        "difficulty": "medium",
    },

    {
        "domain": "smart_building_iot",
        "rough_idea": "An IoT platform to manage energy, security, and comfort systems in a commercial office building.",
        "ground_truth_reqs": [
            "The system shall integrate with HVAC, lighting, and access control sensors via MQTT.",
            "The system shall automatically adjust temperature based on occupancy detected by motion sensors.",
            "The system shall allow building managers to set energy budgets and alert when exceeded.",
            "The system shall provide a floor-plan view showing real-time occupancy per zone.",
            "The system shall control access door locks and log all entry and exit events.",
            "The system shall detect anomalies such as unusual after-hours access and send alerts.",
            "The system shall generate monthly energy consumption reports by floor and system type.",
            "The system shall allow remote control of any connected device from the management dashboard.",
        ],
        "nfr": [
            "Device command latency shall be under 500ms.",
            "System shall support 10,000 concurrent IoT device connections.",
            "All device communications shall use TLS 1.3.",
            "System shall remain operational if internet connectivity is lost (local fallback mode).",
        ],
        "stakeholders": [
            Stakeholder("Ben", "building_manager", "energy cost reduction", "tenant comfort", None),
            Stakeholder("Cleo", "security_officer", "access control audit trail", "intrusion detection", None),
            Stakeholder("Dan", "tenant_company", "comfortable working environment", "privacy from landlord", "building_manager"),
        ],
        "domain_entities": ["Sensor", "Device", "Zone", "OccupancyEvent", "EnergyReading", "AccessEvent", "Alert"],
        "conflicts": [
            {"req_a": "Building manager shall have access to individual occupancy data per desk.",
             "req_b": "Tenant privacy policy prohibits individual-level location tracking of employees.",
             "type": "granularity_privacy_conflict"},
        ],
        "difficulty": "hard",
    },

    # ── SECTOR 6: SOCIAL & PRODUCTIVITY ─────────────────────────────────

    {
        "domain": "task_management_saas",
        "rough_idea": "A project management tool for software teams to track tasks, sprints, and bugs.",
        "ground_truth_reqs": [
            "The system shall allow users to create projects and invite team members with defined roles.",
            "The system shall support Kanban boards and sprint-based backlog management.",
            "The system shall allow tasks to be assigned to team members with due dates and priority levels.",
            "The system shall track time logged against each task.",
            "The system shall generate velocity charts and burndown charts for each sprint.",
            "The system shall provide a GitHub integration to link commits and PRs to tasks.",
            "The system shall send daily digest emails with overdue and at-risk tasks.",
            "The system shall support custom workflow states beyond the default To Do / In Progress / Done.",
        ],
        "nfr": [
            "API shall respond within 200ms at the 99th percentile.",
            "Data shall be exportable in CSV and JSON formats.",
            "System shall support SSO via SAML 2.0 for enterprise customers.",
            "99.9% monthly uptime SLA.",
        ],
        "stakeholders": [
            Stakeholder("Elle", "developer", "minimal process overhead", "clear task ownership", None),
            Stakeholder("Fred", "product_manager", "sprint planning visibility", "stakeholder reporting", None),
            Stakeholder("Gina", "engineering_manager", "team velocity tracking", "resource allocation", None),
        ],
        "domain_entities": ["Project", "Task", "Sprint", "User", "Team", "TimeLog", "GitCommit", "WorkflowState"],
        "conflicts": [
            {"req_a": "Developers shall be able to re-estimate tasks at any time during a sprint.",
             "req_b": "Sprint commitments shall be locked at sprint start to maintain predictability.",
             "type": "agile_methodology_conflict"},
        ],
        "difficulty": "easy",
    },

    {
        "domain": "remote_team_collaboration",
        "rough_idea": "A virtual office platform where distributed teams can have video calls, share screens, and maintain persistent chat rooms.",
        "ground_truth_reqs": [
            "The system shall provide persistent text channels organised by team and topic.",
            "The system shall support video calls for up to 50 participants with screen sharing.",
            "The system shall allow users to set a status indicating availability.",
            "The system shall provide a shared virtual whiteboard for brainstorming sessions.",
            "The system shall allow file sharing up to 500MB per file with preview support.",
            "The system shall record meetings on request and make recordings available for 30 days.",
            "The system shall support threaded replies within channels to keep conversations organised.",
            "The system shall provide a company-wide searchable message archive.",
        ],
        "nfr": [
            "Video calls shall maintain quality at 720p on a 5Mbps connection.",
            "Message delivery latency shall be under 100ms.",
            "End-to-end encryption for direct messages.",
            "GDPR compliance for EU users.",
        ],
        "stakeholders": [
            Stakeholder("Hugo", "remote_employee", "seamless async communication", "work-life boundary", None),
            Stakeholder("Isla", "hr_manager", "culture and engagement monitoring", "compliance", None),
            Stakeholder("Jake", "cto", "security and enterprise features", "integration with existing tools", None),
        ],
        "domain_entities": ["User", "Channel", "Message", "VideoCall", "Recording", "File", "Whiteboard"],
        "conflicts": [
            {"req_a": "HR shall have access to all message history for compliance purposes.",
             "req_b": "Direct messages shall be end-to-end encrypted and inaccessible to employers.",
             "type": "compliance_privacy_conflict"},
        ],
        "difficulty": "medium",
    },

    # ── SECTOR 7: GOVERNMENT & PUBLIC ────────────────────────────────────

    {
        "domain": "citizen_services_portal",
        "rough_idea": "A government portal where citizens can apply for permits, pay taxes, and access public services online.",
        "ground_truth_reqs": [
            "The system shall allow citizens to create a verified digital identity using government ID.",
            "The system shall list all available services with eligibility criteria and required documents.",
            "The system shall allow citizens to submit permit applications and upload supporting documents.",
            "The system shall track application status and notify citizens at each stage.",
            "The system shall allow online payment of council tax, fines, and fees.",
            "The system shall provide a live chat and callback request option for citizen support.",
            "The system shall generate official PDF acknowledgements for every submitted application.",
            "The system shall support Welsh and English languages for Wales-based deployments.",
        ],
        "nfr": [
            "WCAG 2.2 AA accessibility compliance mandatory.",
            "System shall comply with UK Government Digital Service (GDS) standards.",
            "99.99% uptime during business hours.",
            "All data stored in UK data centres.",
        ],
        "stakeholders": [
            Stakeholder("Lena", "citizen", "quick self-service without queuing", "privacy of personal data", None),
            Stakeholder("Mike", "council_caseworker", "structured application data", "manageable caseload", None),
            Stakeholder("Nina", "digital_director", "GDS compliance and cost savings", "accessibility", None),
        ],
        "domain_entities": ["Citizen", "Application", "Service", "Document", "Payment", "CaseWorker", "Notification"],
        "conflicts": [
            {"req_a": "System shall auto-approve low-risk permit applications within 24 hours.",
             "req_b": "All permit approvals shall require manual sign-off by a qualified caseworker.",
             "type": "automation_accountability_conflict"},
        ],
        "difficulty": "hard",
    },

    # ── SECTOR 8: ENTERTAINMENT & MEDIA ──────────────────────────────────

    {
        "domain": "video_streaming",
        "rough_idea": "A video streaming platform for watching movies and TV shows on any device.",
        "ground_truth_reqs": [
            "The system shall provide a content catalogue with search, genre filter, and recommendation.",
            "The system shall stream video in 480p, 720p, 1080p, and 4K based on network speed.",
            "The system shall allow users to download content for offline viewing.",
            "The system shall support user profiles within a single household account.",
            "The system shall track watch history and resume playback from the last position.",
            "The system shall implement parental controls with PIN-protected content ratings.",
            "The system shall provide subtitle support in at least 10 languages.",
            "The system shall support simultaneous streaming on up to 4 devices per account.",
        ],
        "nfr": [
            "Video shall start within 3 seconds on a 10Mbps connection.",
            "DRM protection via Widevine or FairPlay.",
            "CDN shall serve content from the nearest edge node.",
            "System shall handle 1 million concurrent streams.",
        ],
        "stakeholders": [
            Stakeholder("Ora", "subscriber", "wide content library and offline access", "affordable pricing", None),
            Stakeholder("Pete", "content_partner", "DRM enforcement and royalty tracking", "audience analytics", None),
            Stakeholder("Rita", "product_manager", "subscriber growth and retention", "personalisation accuracy", None),
        ],
        "domain_entities": ["Content", "User", "Profile", "WatchHistory", "Download", "Subscription", "Recommendation"],
        "conflicts": [
            {"req_a": "Downloads shall be available for unlimited offline viewing.",
             "req_b": "Content licences restrict offline viewing to 30 days and 5 saves per title.",
             "type": "licence_constraint_conflict"},
        ],
        "difficulty": "medium",
    },

    # Add more templates here as your dataset grows ...
]


# ─────────────────────────────────────────────
#  ScenarioGenerator
# ─────────────────────────────────────────────

class ScenarioGenerator:
    """
    Generates synthetic RE scenarios for training REMARL agents.

    Usage:
        gen = ScenarioGenerator("data/scenarios/")
        scenario = gen.sample()              # random scenario
        scenario = gen.sample(domain="healthcare")  # domain-specific
        scenario = gen.sample(difficulty="hard")    # difficulty-specific
        batch    = gen.sample_batch(n=32)    # batch for training

    The scenario_dir is used to cache generated scenarios as JSON files.
    On first run, all templates are expanded and cached.
    On subsequent runs, cached scenarios are loaded for reproducibility.
    """

    def __init__(
        self,
        scenario_dir: str = "data/scenarios/",
        hide_fraction: float = 0.25,   # fraction of reqs to hide per episode
        seed: int = 42,
    ):
        self.scenario_dir = pathlib.Path(scenario_dir)
        self.scenario_dir.mkdir(parents=True, exist_ok=True)
        self.hide_fraction = hide_fraction
        self.rng = random.Random(seed)

        self._templates = DOMAIN_TEMPLATES
        self._scenarios: List[Scenario] = []
        self._load_or_build()

        logger.info(
            f"ScenarioGenerator ready: {len(self._scenarios)} scenarios "
            f"across {len(set(s.domain for s in self._scenarios))} domains."
        )

    # ── public API ───────────────────────────────────────────────────────

    def sample(
        self,
        domain: Optional[str] = None,
        difficulty: Optional[str] = None,
    ) -> Scenario:
        """Sample one scenario, optionally filtered."""
        pool = self._filter(domain, difficulty)
        if not pool:
            raise ValueError(
                f"No scenarios match domain={domain}, difficulty={difficulty}. "
                f"Available domains: {self.available_domains()}"
            )
        scenario = self.rng.choice(pool)
        # Reshuffle which reqs are hidden each call — adds training variance
        return self._apply_hiding(scenario)

    def sample_batch(
        self,
        n: int,
        domain: Optional[str] = None,
        difficulty: Optional[str] = None,
    ) -> List[Scenario]:
        """Sample n scenarios (with replacement)."""
        return [self.sample(domain=domain, difficulty=difficulty) for _ in range(n)]

    def available_domains(self) -> List[str]:
        return sorted(set(s.domain for s in self._scenarios))

    def available_difficulties(self) -> List[str]:
        return sorted(set(s.difficulty for s in self._scenarios))

    def stats(self) -> dict:
        """Summary statistics for logging."""
        from collections import Counter
        return {
            "total": len(self._scenarios),
            "by_domain": dict(Counter(s.domain for s in self._scenarios)),
            "by_difficulty": dict(Counter(s.difficulty for s in self._scenarios)),
            "avg_reqs": sum(len(s.ground_truth_reqs) for s in self._scenarios) / len(self._scenarios),
            "avg_hidden": sum(len(s.hidden_reqs) for s in self._scenarios) / len(self._scenarios),
        }

    # ── internal helpers ─────────────────────────────────────────────────

    def _load_or_build(self):
        cache_file = self.scenario_dir / "all_scenarios.json"
        if cache_file.exists():
            with open(cache_file) as f:
                raw = json.load(f)
            self._scenarios = [Scenario.from_dict(d) for d in raw]
            logger.info(f"Loaded {len(self._scenarios)} cached scenarios.")
        else:
            self._build_and_cache(cache_file)

    def _build_and_cache(self, cache_file: pathlib.Path):
        """Convert templates → Scenario objects and persist."""
        for t in self._templates:
            scenario = self._template_to_scenario(t)
            self._scenarios.append(scenario)

        with open(cache_file, "w") as f:
            json.dump([s.to_dict() for s in self._scenarios], f, indent=2)
        logger.info(f"Built and cached {len(self._scenarios)} scenarios.")

    def _template_to_scenario(self, t: dict) -> Scenario:
        reqs = t["ground_truth_reqs"]
        n_hide = max(1, int(len(reqs) * self.hide_fraction))
        hidden = self.rng.sample(reqs, n_hide)
        visible = [r for r in reqs if r not in hidden]

        # Deterministic scenario ID from content hash
        content = t["domain"] + t["rough_idea"]
        sid = hashlib.md5(content.encode()).hexdigest()[:8]

        stakeholders = t.get("stakeholders", [])
        # Convert to Stakeholder objects if they're dicts
        if stakeholders and isinstance(stakeholders[0], dict):
            stakeholders = [Stakeholder(**s) for s in stakeholders]

        return Scenario(
            scenario_id=sid,
            domain=t["domain"],
            rough_idea=t["rough_idea"],
            ground_truth_reqs=reqs,
            hidden_reqs=hidden,
            visible_reqs=visible,
            nfr=t.get("nfr", []),
            stakeholders=stakeholders,
            domain_entities=t.get("domain_entities", []),
            conflicts=t.get("conflicts", []),
            difficulty=t.get("difficulty", "medium"),
        )

    def _apply_hiding(self, scenario: Scenario) -> Scenario:
        """Re-randomise which reqs are hidden (new each call)."""
        reqs = scenario.ground_truth_reqs
        n_hide = max(1, int(len(reqs) * self.hide_fraction))
        hidden = self.rng.sample(reqs, n_hide)
        visible = [r for r in reqs if r not in hidden]
        # Return a new Scenario with updated hidden/visible split
        import copy
        s = copy.deepcopy(scenario)
        s.hidden_reqs = hidden
        s.visible_reqs = visible
        return s

    def _filter(
        self,
        domain: Optional[str],
        difficulty: Optional[str],
    ) -> List[Scenario]:
        pool = self._scenarios
        if domain:
            pool = [s for s in pool if s.domain == domain]
        if difficulty:
            pool = [s for s in pool if s.difficulty == difficulty]
        return pool


# ─────────────────────────────────────────────
#  Quick smoke test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    gen = ScenarioGenerator("data/scenarios/")
    print("\n── Stats ──")
    import pprint; pprint.pprint(gen.stats())

    print("\n── Sample scenario ──")
    s = gen.sample()
    print(f"Domain   : {s.domain}")
    print(f"Idea     : {s.rough_idea[:80]}...")
    print(f"Total FR : {len(s.ground_truth_reqs)}")
    print(f"Visible  : {len(s.visible_reqs)}  |  Hidden: {len(s.hidden_reqs)}")
    print(f"NFR      : {len(s.nfr)}")
    print(f"Conflict : {len(s.conflicts)}")
    print(f"Difficulty: {s.difficulty}")
    print(f"\nHidden reqs the agents must discover:")
    for r in s.hidden_reqs:
        print(f"  - {r}")
