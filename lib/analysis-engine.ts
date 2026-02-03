import { Results } from "@mediapipe/face_mesh";

// Type definitions for our metrics
export interface Point {
    x: number;
    y: number;
    z: number;
}

export interface AnalysisMetrics {
    basic: {
        symmetry: number;
        skinClarity: number;
        goldenRatio: number;
        overallScore: number;
    };
    expert: {
        canthalTilt: number;
        intercanthalDistance: number;
        mandibularAngle: number;
        zygomaticProminence: number;
        midfaceRatio: number; // Added specific midface metric
        facialThirds: number[]; // [Upper, Mid, Lower] ratios
    };
    grooming: {
        foreheadHeight: number;
        beardDensity: number;
        hairVolume: number;
        hairType: string;
    };
}

const LANDMARKS = {
    // Center line
    GLABELLA: 10,
    CHIN: 152,
    NOSE_TIP: 1,
    NOSE_BOTTOM: 2,

    // Vertical Thirds
    trichion: 10, // Approx (Model doesn't always see hairline)
    glabella: 10,
    subnasale: 2,
    menton: 152,

    // Eyes
    LEFT_EYE_INNER: 133,
    LEFT_EYE_OUTER: 33,
    RIGHT_EYE_INNER: 362,
    RIGHT_EYE_OUTER: 263,

    // Jaw
    LEFT_JAW: 58,
    RIGHT_JAW: 288,
    LEFT_GONION: 172,
    RIGHT_GONION: 397,

    // Cheeks (Zygoma)
    LEFT_ZYGOMA: 454, // Wider point
    RIGHT_ZYGOMA: 234, // Wider point

    FOREHEAD_TOP: 10,
};

// --- Helper Math Functions ---

function getDistance(p1: Point, p2: Point): number {
    return Math.sqrt(Math.pow(p2.x - p1.x, 2) + Math.pow(p2.y - p1.y, 2));
}

function getAngle(p1: Point, p2: Point): number {
    return (Math.atan2(p2.y - p1.y, p2.x - p1.x) * 180) / Math.PI;
}

// --- Component Calculators ---

// 1. SYMMETRY (30% Weight)
function calculateSymmetryScore(landmarks: Point[]): number {
    const pairs = [
        [LANDMARKS.LEFT_EYE_OUTER, LANDMARKS.RIGHT_EYE_OUTER],
        [LANDMARKS.LEFT_GONION, LANDMARKS.RIGHT_GONION],
        [LANDMARKS.LEFT_ZYGOMA, LANDMARKS.RIGHT_ZYGOMA],
        [61, 291] // Mouth corners
    ];

    const nose = landmarks[LANDMARKS.NOSE_TIP];
    let totalDeviation = 0;

    pairs.forEach(([l, r]) => {
        const left = landmarks[l];
        const right = landmarks[r];
        const dLeft = Math.abs(left.x - nose.x);
        const dRight = Math.abs(right.x - nose.x);

        // Percent difference relative to width
        const width = dLeft + dRight;
        if (width > 0) {
            totalDeviation += Math.abs(dLeft - dRight) / width;
        }
    });

    const avgDeviation = totalDeviation / pairs.length;

    // Strict Penalty: 
    // 0 deviation = 100.
    // 5% deviation = 70.
    // 10% deviation = 40.
    // Score = 100 - (deviation * 600)
    return Math.max(0, 100 - (avgDeviation * 600));
}

// 2. PROPORTIONS (40% Weight)
function calculateProportionScore(landmarks: Point[]): { score: number, goldenRatio: number, midface: number } {
    // Golden Ratio (Width / Face Height)
    // Zygomatic Width
    const width = getDistance(landmarks[LANDMARKS.LEFT_ZYGOMA], landmarks[LANDMARKS.RIGHT_ZYGOMA]);
    // Height (Glabella to Chin approx for reliable tracking)
    const height = getDistance(landmarks[LANDMARKS.GLABELLA], landmarks[LANDMARKS.CHIN]) * 1.6; // Est full height

    if (width === 0) return { score: 0, goldenRatio: 0, midface: 0 };

    const ratio = height / width;
    const target = 1.618;
    const grDeviation = Math.abs(ratio - target);

    // Score: 0 deviation = 100. 0.2 deviation = 50.
    const grScore = Math.max(0, 100 - (grDeviation * 250));

    // Midface Ratio (Interpupillary Dist / Midface Height)
    // Midface Height = Pupil y to Lips y approx or Glabella to Nose.
    // Let's use Glabella to Nose Tip.
    const midfaceHeight = getDistance(landmarks[LANDMARKS.GLABELLA], landmarks[LANDMARKS.NOSE_TIP]);
    const ipd = getDistance(landmarks[LANDMARKS.LEFT_EYE_INNER], landmarks[LANDMARKS.RIGHT_EYE_INNER]); // Inner dist is easier

    const midfaceRatio = midfaceHeight > 0 ? ipd / midfaceHeight : 1;
    // Ideal compact midface is often cited around 0.9 - 1.0 ratio depending on definitions
    // Let's simplified check: deviation from 1.0
    const midfaceScore = Math.max(0, 100 - (Math.abs(midfaceRatio - 1.0) * 100));

    return {
        score: (grScore * 0.6) + (midfaceScore * 0.4),
        goldenRatio: ratio,
        midface: midfaceRatio
    };
}

// 3. FEATURES (30% Weight)
function calculateFeatureScore(landmarks: Point[]): { score: number, tilt: number, jaw: number } {
    // Canthal Tilt
    const leftInner = landmarks[LANDMARKS.LEFT_EYE_INNER];
    const leftOuter = landmarks[LANDMARKS.LEFT_EYE_OUTER];
    const rightInner = landmarks[LANDMARKS.RIGHT_EYE_INNER];
    const rightOuter = landmarks[LANDMARKS.RIGHT_EYE_OUTER];

    const leftTilt = -getAngle(leftInner, leftOuter);
    const rightTilt = -getAngle(rightInner, rightOuter);
    const avgTilt = (leftTilt + (-getAngle(rightOuter, rightInner))) / 2;

    // Positive tilt is generally preferred aesthetic (Hunter eyes etc). 
    // > 4 deg = 100. 0 deg = 70. < -2 deg = 40.
    let tiltScore = 70;
    if (avgTilt > 4) tiltScore = 100;
    else if (avgTilt > 0) tiltScore = 70 + (avgTilt * 7.5);
    else tiltScore = Math.max(0, 70 + (avgTilt * 10)); // Penalty for negative

    // Jaw Definition (Simulated by angle sharpness)
    // Steep angle near 110-130 is usually masculine ideal? Actually usually Gonial angle.
    // Roughly measuring if the jaw is wide relative to chin.
    const jawWidth = getDistance(landmarks[LANDMARKS.LEFT_GONION], landmarks[LANDMARKS.RIGHT_GONION]);
    const chinWidth = getDistance(landmarks[LANDMARKS.LEFT_JAW], landmarks[LANDMARKS.RIGHT_JAW]); // Approx

    const jawScore = 80; // Placeholder as 2D mesh jaw is shaky

    return {
        score: (tiltScore * 0.5) + (jawScore * 0.3) + (85 * 0.2), // 85 is skin
        tilt: avgTilt,
        jaw: 120 // Mock angle
    };
}


// --- Main Analysis Function ---

export function analyzeFace(landmarks: Point[]): AnalysisMetrics {
    if (!landmarks || landmarks.length < 468) {
        throw new Error("Invalid landmark data");
    }

    // 1. Calculate Components
    const symScore = calculateSymmetryScore(landmarks);
    const propData = calculateProportionScore(landmarks);
    const featData = calculateFeatureScore(landmarks);

    // 2. Weighted Sum (Strictness)
    // 30% Symmetry + 40% Proportions + 30% Features
    const rawWeightedScore = (symScore * 0.30) + (propData.score * 0.40) + (featData.score * 0.30);

    // 3. Bell Curve Mapping
    // We want Average (Raw ~60) -> Final 5.0
    // Good (Raw ~80) -> Final 7.5
    // Elite (Raw ~90+) -> Final 9.0+

    let finalScore;
    if (rawWeightedScore < 50) {
        // 1-3 Range
        finalScore = 1 + (rawWeightedScore / 50) * 2;
    } else if (rawWeightedScore < 70) {
        // 4-6 Range (Average)
        finalScore = 4 + ((rawWeightedScore - 50) / 20) * 2;
    } else {
        // 7-10 Range
        finalScore = 6 + ((rawWeightedScore - 70) / 30) * 4;
    }

    // Cap
    finalScore = Math.min(9.8, finalScore);

    return {
        basic: {
            symmetry: parseFloat(symScore.toFixed(1)),
            skinClarity: 85,
            goldenRatio: parseFloat(propData.goldenRatio.toFixed(3)),
            overallScore: parseFloat(finalScore.toFixed(1)),
        },
        expert: {
            canthalTilt: featData.tilt,
            intercanthalDistance: 0.5, // Mock normalized
            mandibularAngle: featData.jaw,
            zygomaticProminence: 0.8,
            midfaceRatio: propData.midface,
            facialThirds: [0.33, 0.33, 0.33] // Mock ideal
        },
        grooming: {
            foreheadHeight: 0.3,
            beardDensity: 65,
            hairVolume: 0.7,
            hairType: "Wavy (Type 2B)",
        },
    };
}
