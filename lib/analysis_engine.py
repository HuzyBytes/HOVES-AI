"""
HOVES Facial Analysis Engine
Port from TypeScript (lib/analysis-engine.ts)
Calculates facial metrics from MediaPipe landmarks
"""

import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple

@dataclass
class Point:
    """3D point representation"""
    x: float
    y: float
    z: float

@dataclass
class BasicMetrics:
    symmetry: float
    skin_clarity: int
    golden_ratio: float
    overall_score: float

@dataclass
class ExpertMetrics:
    canthal_tilt: float
    intercanthal_distance: float
    mandibular_angle: float
    zygomatic_prominence: float
    midface_ratio: float
    facial_thirds: List[float]

@dataclass
class GroomingMetrics:
    forehead_height: float
    beard_density: int
    hair_volume: float
    hair_type: str

@dataclass
class Insights:
    fun_facts: List[str]
    recommendations: List[str]
    routine_guide: List[str]

@dataclass
class AnalysisMetrics:
    basic: BasicMetrics
    expert: ExpertMetrics
    grooming: GroomingMetrics
    insights: Insights
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            'basic': asdict(self.basic),
            'expert': asdict(self.expert),
            'grooming': asdict(self.grooming),
            'insights': asdict(self.insights)
        }

def generate_insights(basic: BasicMetrics, expert: ExpertMetrics, grooming: GroomingMetrics) -> Insights:
    """Generate personalized fun facts, recommendations, and routines based on metrics"""
    
    fun_facts = []
    recommendations = []
    routine = []

    # Fun Facts based on biometrics
    if basic.symmetry > 95:
        fun_facts.append("Your facial symmetry is in the top 1% of the population, often linked to high genetic health.")
    elif basic.symmetry > 90:
        fun_facts.append("High facial symmetry like yours is historically associated with perceived trustworthiness.")
    
    if expert.canthal_tilt > 5:
        fun_facts.append("Your 'positive canthal tilt' is a trait shared by many high-fashion models, giving an alert, youthful appearance.")
    elif expert.canthal_tilt < 0:
        fun_facts.append("A neutral/negative canthal tilt often creates a 'relaxed' or 'dreamy' look, similar to many classic Hollywood actors.")

    if basic.overall_score > 7.5:
        fun_facts.append("Your facial proportions align significantly with the 'Golden Ratio' used by Renaissance artists.")

    # Recommendations based on weaknesses
    if basic.skin_clarity < 85:
        recommendations.append("Prioritize niacinamide or Vitamin C serums to improve skin texture and metric scores.")
    
    if expert.mandibular_angle > 130:
        recommendations.append("Consider masseter exercises or 'mewing' techniques to sharpen jawline definition.")
    
    if grooming.beard_density < 70 and grooming.beard_density > 30:
        recommendations.append("A shorter, well-groomed 'stubble' look would emphasize your jawline better than a full beard.")
    
    if expert.midface_ratio > 1.05:
        recommendations.append("Hairstyles with more volume on the sides can balance a slightly longer midface ratio.")

    # Routine Guide
    routine.append("MORNING: Apply SPF 50+ daily to preserve skin clarity metrics.")
    
    if basic.skin_clarity < 80:
        routine.append("EVENING: Double cleanse followed by a gentle retinoid to boost clarity score.")
    else:
        routine.append("EVENING: Hydrating mask once a week to maintain your high clarity index.")

    if expert.mandibular_angle > 125:
        routine.append("EXERCISE: 5 minutes of facial yoga targeting the lower third to maintain mandibular definition.")

    # Fallbacks if list is too short
    if not fun_facts: fun_facts = ["Your facial structure follows unique biometric patterns not commonly seen in standard datasets."]
    if not recommendations: recommendations = ["Focus on consistent sleep (7-9h) to maintain current aesthetic metrics."]
    if not routine: routine = ["Standard: Cleanse, Moisturize, Protect. Consistency is key for biometric stability."]

    return Insights(
        fun_facts=fun_facts[:3],
        recommendations=recommendations[:3],
        routine_guide=routine[:4]
    )

# Landmark indices (same as TypeScript)
class LANDMARKS:
    # Center line
    GLABELLA = 10
    CHIN = 152
    NOSE_TIP = 1
    NOSE_BOTTOM = 2
    
    # Vertical Thirds
    TRICHION = 10
    SUBNASALE = 2
    MENTON = 152
    
    # Eyes
    LEFT_EYE_INNER = 133
    LEFT_EYE_OUTER = 33
    RIGHT_EYE_INNER = 362
    RIGHT_EYE_OUTER = 263
    
    # Jaw
    LEFT_JAW = 58
    RIGHT_JAW = 288
    LEFT_GONION = 172
    RIGHT_GONION = 397
    
    # Cheeks (Zygoma)
    LEFT_ZYGOMA = 454
    RIGHT_ZYGOMA = 234
    
    FOREHEAD_TOP = 10


def get_distance(p1: Point, p2: Point) -> float:
    """Calculate Euclidean distance between two points"""
    return np.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2)


def get_angle(p1: Point, p2: Point) -> float:
    """Calculate angle in degrees between two points"""
    return np.degrees(np.arctan2(p2.y - p1.y, p2.x - p1.x))


def calculate_symmetry_score(landmarks: List[Point]) -> float:
    """
    Calculate facial symmetry score (30% weight)
    Returns: 0-100 score
    """
    pairs = [
        (LANDMARKS.LEFT_EYE_OUTER, LANDMARKS.RIGHT_EYE_OUTER),
        (LANDMARKS.LEFT_GONION, LANDMARKS.RIGHT_GONION),
        (LANDMARKS.LEFT_ZYGOMA, LANDMARKS.RIGHT_ZYGOMA),
        (61, 291)  # Mouth corners
    ]
    
    nose = landmarks[LANDMARKS.NOSE_TIP]
    total_deviation = 0.0
    
    for left_idx, right_idx in pairs:
        left = landmarks[left_idx]
        right = landmarks[right_idx]
        d_left = abs(left.x - nose.x)
        d_right = abs(right.x - nose.x)
        
        # Percent difference relative to width
        width = d_left + d_right
        if width > 0:
            total_deviation += abs(d_left - d_right) / width
    
    avg_deviation = total_deviation / len(pairs)
    
    # Strict Penalty (same as TypeScript)
    # 0 deviation = 100, 5% deviation = 70, 10% deviation = 40
    return max(0, 100 - (avg_deviation * 600))


def calculate_proportion_score(landmarks: List[Point]) -> Tuple[float, float, float]:
    """
    Calculate proportions score (40% weight)
    Returns: (score, golden_ratio, midface_ratio)
    """
    # Golden Ratio (Width / Face Height)
    width = get_distance(landmarks[LANDMARKS.LEFT_ZYGOMA], landmarks[LANDMARKS.RIGHT_ZYGOMA])
    height = get_distance(landmarks[LANDMARKS.GLABELLA], landmarks[LANDMARKS.CHIN]) * 1.6  # Estimate full height
    
    if width == 0:
        return 0.0, 0.0, 0.0
    
    ratio = height / width
    target = 1.618
    gr_deviation = abs(ratio - target)
    
    # Score: 0 deviation = 100, 0.2 deviation = 50
    gr_score = max(0, 100 - (gr_deviation * 250))
    
    # Midface Ratio
    midface_height = get_distance(landmarks[LANDMARKS.GLABELLA], landmarks[LANDMARKS.NOSE_TIP])
    ipd = get_distance(landmarks[LANDMARKS.LEFT_EYE_INNER], landmarks[LANDMARKS.RIGHT_EYE_INNER])
    
    midface_ratio = ipd / midface_height if midface_height > 0 else 1.0
    midface_score = max(0, 100 - (abs(midface_ratio - 1.0) * 100))
    
    final_score = (gr_score * 0.6) + (midface_score * 0.4)
    
    return final_score, ratio, midface_ratio


def calculate_feature_score(landmarks: List[Point], symmetry_score: float) -> Tuple[float, float, float]:
    """
    Calculate feature quality score (30% weight)
    Returns: (score, canthal_tilt, jaw_angle)
    """
    # Canthal Tilt
    left_inner = landmarks[LANDMARKS.LEFT_EYE_INNER]
    left_outer = landmarks[LANDMARKS.LEFT_EYE_OUTER]
    right_inner = landmarks[LANDMARKS.RIGHT_EYE_INNER]
    right_outer = landmarks[LANDMARKS.RIGHT_EYE_OUTER]
    
    left_tilt = -get_angle(left_inner, left_outer)
    right_tilt = -get_angle(right_inner, right_outer)
    avg_tilt = (left_tilt + (-get_angle(right_outer, right_inner))) / 2
    
    # Positive tilt is preferred (Hunter eyes)
    # > 4 deg = 100, 0 deg = 70, < -2 deg = 40
    if avg_tilt > 4:
        tilt_score = 100
    elif avg_tilt > 0:
        tilt_score = 70 + (avg_tilt * 7.5)
    else:
        tilt_score = max(0, 70 + (avg_tilt * 10))
    
    # Jaw Definition (Mandibular Angle)
    # 2D estimation using GO-ME and GO-GL indices
    left_gonion = landmarks[LANDMARKS.LEFT_GONION]
    left_jaw = landmarks[LANDMARKS.LEFT_JAW]
    right_gonion = landmarks[LANDMARKS.RIGHT_GONION]
    right_jaw = landmarks[LANDMARKS.RIGHT_JAW]
    
    # Mandibular angle estimate: angle between jawline and ramus
    # This is a Rough 2D approximation
    left_jaw_angle = abs(get_angle(left_gonion, left_jaw) - get_angle(left_gonion, landmarks[LANDMARKS.GLABELLA]))
    right_jaw_angle = abs(get_angle(right_gonion, right_jaw) - get_angle(right_gonion, landmarks[LANDMARKS.GLABELLA]))
    avg_jaw_angle = (left_jaw_angle + right_jaw_angle) / 2
    
    # Ideal mandibular angle for men is ~120-130 deg
    jaw_score = max(0, 100 - (abs(avg_jaw_angle - 125) * 2))
    
    # Skin Clarity Estimate (Higher base)
    skin_score = 85 + (100 - symmetry_score) * 0.05
    
    final_score = (tilt_score * 0.45) + (jaw_score * 0.35) + (skin_score * 0.2)
    
    return final_score, avg_tilt, avg_jaw_angle, skin_score


def analyze_face(landmarks_data: List[Dict]) -> AnalysisMetrics:
    """Main analysis function with insight generation"""
    if not landmarks_data or len(landmarks_data) < 468:
        raise ValueError("Invalid landmark data: Expected 468 landmarks")
    
    # Convert to Point objects
    landmarks = [Point(x=lm['x'], y=lm['y'], z=lm.get('z', 0)) for lm in landmarks_data]
    
    # Calculate component scores
    sym_score = calculate_symmetry_score(landmarks)
    prop_score, golden_ratio, midface_ratio = calculate_proportion_score(landmarks)
    feat_score, canthal_tilt, jaw_angle, skin_score = calculate_feature_score(landmarks, sym_score)
    
    # Intercanthal Distance
    face_width = get_distance(landmarks[LANDMARKS.LEFT_ZYGOMA], landmarks[LANDMARKS.RIGHT_ZYGOMA])
    icd = get_distance(landmarks[LANDMARKS.LEFT_EYE_INNER], landmarks[LANDMARKS.RIGHT_EYE_INNER])
    icd_ratio = icd / face_width if face_width > 0 else 0.5
    
    # Facial Thirds
    forehead = get_distance(landmarks[LANDMARKS.TRICHION], landmarks[LANDMARKS.GLABELLA])
    midface = get_distance(landmarks[LANDMARKS.GLABELLA], landmarks[LANDMARKS.SUBNASALE])
    lowerface = get_distance(landmarks[LANDMARKS.SUBNASALE], landmarks[LANDMARKS.MENTON])
    total_h = forehead + midface + lowerface
    thirds = [forehead/total_h, midface/total_h, lowerface/total_h] if total_h > 0 else [0.33, 0.33, 0.33]
    
    # Bell Curve Mapping (Shifted for better user satisfaction)
    raw_weighted_score = (sym_score * 0.30) + (prop_score * 0.40) + (feat_score * 0.30)
    
    if raw_weighted_score < 40:
        # 1-5 Range for lower scores
        final_score = 1 + (raw_weighted_score / 40) * 4
    elif raw_weighted_score < 70:
        # 5-8 Range (Good/Above Average)
        final_score = 5 + ((raw_weighted_score - 40) / 30) * 3
    else:
        # 8-10 Range (Elite/Perfect)
        final_score = 8 + ((raw_weighted_score - 70) / 30) * 2
    
    # Cap at 9.9 for high performers
    final_score = min(9.9, final_score)
    
    # Build metrics
    basic=BasicMetrics(
        symmetry=round(sym_score, 1),
        skin_clarity=int(skin_score),
        golden_ratio=round(golden_ratio, 3),
        overall_score=round(final_score, 1)
    )
    expert=ExpertMetrics(
        canthal_tilt=round(canthal_tilt, 1),
        intercanthal_distance=round(icd_ratio, 3),
        mandibular_angle=round(jaw_angle, 1),
        zygomatic_prominence=round(0.7 + (sym_score/1000), 2),
        midface_ratio=round(midface_ratio, 3),
        facial_thirds=[round(t, 3) for t in thirds]
    )
    grooming=GroomingMetrics(
        forehead_height=round(thirds[0], 3),
        beard_density=int(60 + (sym_score/10)),
        hair_volume=round(0.6 + (sym_score/250), 2),
        hair_type="Wavy (Type 2B)"
    )
    
    # Generate insights
    insights = generate_insights(basic, expert, grooming)
    
    return AnalysisMetrics(
        basic=basic,
        expert=expert,
        grooming=grooming,
        insights=insights
    )


if __name__ == '__main__':
    # Test with mock data
    print("Testing analysis engine...")
    
    # Create mock landmarks (468 points)
    mock_landmarks = [{'x': 0.5, 'y': 0.5, 'z': 0} for _ in range(468)]
    
    try:
        metrics = analyze_face(mock_landmarks)
        print("✓ Analysis successful")
        print(f"  Overall Score: {metrics.basic.overall_score}/10")
        print(f"  Symmetry: {metrics.basic.symmetry}%")
        print(f"  Golden Ratio: {metrics.basic.golden_ratio}")
    except Exception as e:
        print(f"✗ Analysis failed: {e}")
