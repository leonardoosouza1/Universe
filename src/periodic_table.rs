use bevy::prelude::*;

pub struct ElementData {
    pub name: &'static str,
    pub symbol: &'static str,
    pub mass: f32,
    pub electronegativity: f32,
    pub radius: f32,
    pub color: Color,
}


//NOT CHANGE OR REMOVE ELEMENTS
pub const PERIODIC_TABLE: &[ElementData] = &[
    // 1-2: Período 1
    ElementData { name: "Hydrogen", symbol: "H", mass: 1.008, electronegativity: 2.20, radius: 37.0, color: Color::srgb(0.7, 0.7, 1.0) },
    ElementData { name: "Helium", symbol: "He", mass: 4.002, electronegativity: 0.00, radius: 31.0, color: Color::srgb(0.8, 1.0, 1.0) },
    
    // 3-10: Período 2
    ElementData { name: "Lithium", symbol: "Li", mass: 6.941, electronegativity: 0.98, radius: 152.0, color: Color::srgb(0.8, 0.3, 0.6) },
    ElementData { name: "Beryllium", symbol: "Be", mass: 9.012, electronegativity: 1.57, radius: 112.0, color: Color::srgb(0.5, 0.7, 0.5) },
    ElementData { name: "Boron", symbol: "B", mass: 10.811, electronegativity: 2.04, radius: 88.0, color: Color::srgb(1.0, 0.8, 0.7) },
    ElementData { name: "Carbon", symbol: "C", mass: 12.011, electronegativity: 2.55, radius: 77.0, color: Color::srgb(0.2, 0.2, 0.2) },
    ElementData { name: "Nitrogen", symbol: "N", mass: 14.007, electronegativity: 3.04, radius: 70.0, color: Color::srgb(0.5, 0.5, 1.0) },
    ElementData { name: "Oxygen", symbol: "O", mass: 15.999, electronegativity: 3.44, radius: 66.0, color: Color::srgb(1.0, 0.2, 0.2) },
    ElementData { name: "Fluorine", symbol: "F", mass: 18.998, electronegativity: 3.98, radius: 64.0, color: Color::srgb(0.3, 1.0, 0.4) },
    ElementData { name: "Neon", symbol: "Ne", mass: 20.180, electronegativity: 0.00, radius: 58.0, color: Color::srgb(1.0, 0.5, 0.0) },
    
    // 11-18: Período 3
    ElementData { name: "Sodium", symbol: "Na", mass: 22.990, electronegativity: 0.93, radius: 186.0, color: Color::srgb(0.5, 0.2, 0.8) },
    ElementData { name: "Magnesium", symbol: "Mg", mass: 24.305, electronegativity: 1.31, radius: 160.0, color: Color::srgb(0.6, 0.6, 0.2) },
    ElementData { name: "Aluminum", symbol: "Al", mass: 26.982, electronegativity: 1.61, radius: 143.0, color: Color::srgb(0.7, 0.7, 0.7) },
    ElementData { name: "Silicon", symbol: "Si", mass: 28.085, electronegativity: 1.90, radius: 118.0, color: Color::srgb(0.4, 0.4, 0.6) },
    ElementData { name: "Phosphorus", symbol: "P", mass: 30.974, electronegativity: 2.19, radius: 110.0, color: Color::srgb(1.0, 0.6, 0.0) },
    ElementData { name: "Sulfur", symbol: "S", mass: 32.065, electronegativity: 2.58, radius: 104.0, color: Color::srgb(1.0, 1.0, 0.2) },
    ElementData { name: "Chlorine", symbol: "Cl", mass: 35.453, electronegativity: 3.16, radius: 99.0, color: Color::srgb(0.1, 0.8, 0.1) },
    ElementData { name: "Argon", symbol: "Ar", mass: 39.948, electronegativity: 0.00, radius: 94.0, color: Color::srgb(1.0, 0.2, 1.0) },
    
    // 19-36: Período 4
    ElementData { name: "Potassium", symbol: "K", mass: 39.098, electronegativity: 0.82, radius: 227.0, color: Color::srgb(0.8, 0.1, 0.1) },
    ElementData { name: "Calcium", symbol: "Ca", mass: 40.078, electronegativity: 1.00, radius: 197.0, color: Color::srgb(0.6, 0.4, 0.1) },
    ElementData { name: "Scandium", symbol: "Sc", mass: 44.956, electronegativity: 1.36, radius: 162.0, color: Color::srgb(0.9, 0.9, 0.9) },
    ElementData { name: "Titanium", symbol: "Ti", mass: 47.867, electronegativity: 1.54, radius: 147.0, color: Color::srgb(0.7, 0.7, 0.7) },
    ElementData { name: "Vanadium", symbol: "V", mass: 50.942, electronegativity: 1.63, radius: 134.0, color: Color::srgb(0.6, 0.6, 0.6) },
    ElementData { name: "Chromium", symbol: "Cr", mass: 51.996, electronegativity: 1.66, radius: 128.0, color: Color::srgb(0.5, 0.6, 0.8) },
    ElementData { name: "Manganese", symbol: "Mn", mass: 54.938, electronegativity: 1.55, radius: 127.0, color: Color::srgb(0.6, 0.5, 0.7) },
    ElementData { name: "Iron", symbol: "Fe", mass: 55.845, electronegativity: 1.83, radius: 126.0, color: Color::srgb(0.8, 0.4, 0.2) },
    ElementData { name: "Cobalt", symbol: "Co", mass: 58.933, electronegativity: 1.88, radius: 125.0, color: Color::srgb(0.2, 0.4, 0.8) },
    ElementData { name: "Nickel", symbol: "Ni", mass: 58.693, electronegativity: 1.91, radius: 124.0, color: Color::srgb(0.3, 0.7, 0.5) },
    ElementData { name: "Copper", symbol: "Cu", mass: 63.546, electronegativity: 1.90, radius: 128.0, color: Color::srgb(0.9, 0.5, 0.3) },
    ElementData { name: "Zinc", symbol: "Zn", mass: 65.380, electronegativity: 1.65, radius: 134.0, color: Color::srgb(0.5, 0.5, 0.6) },
    ElementData { name: "Gallium", symbol: "Ga", mass: 69.723, electronegativity: 1.81, radius: 135.0, color: Color::srgb(0.7, 0.6, 0.6) },
    ElementData { name: "Germanium", symbol: "Ge", mass: 72.630, electronegativity: 2.01, radius: 122.0, color: Color::srgb(0.4, 0.6, 0.4) },
    ElementData { name: "Arsenic", symbol: "As", mass: 74.922, electronegativity: 2.18, radius: 115.0, color: Color::srgb(0.7, 0.5, 0.8) },
    ElementData { name: "Selenium", symbol: "Se", mass: 78.971, electronegativity: 2.55, radius: 103.0, color: Color::srgb(1.0, 0.6, 0.0) },
    ElementData { name: "Bromine", symbol: "Br", mass: 79.904, electronegativity: 2.96, radius: 114.0, color: Color::srgb(0.6, 0.2, 0.2) },
    ElementData { name: "Krypton", symbol: "Kr", mass: 83.798, electronegativity: 3.00, radius: 109.0, color: Color::srgb(0.4, 0.8, 0.8) },

    // 37-54: Período 5
    ElementData { name: "Rubidium", symbol: "Rb", mass: 85.468, electronegativity: 0.82, radius: 248.0, color: Color::srgb(0.4, 0.2, 0.6) },
    ElementData { name: "Strontium", symbol: "Sr", mass: 87.620, electronegativity: 0.95, radius: 215.0, color: Color::srgb(0.0, 0.8, 0.0) },
    ElementData { name: "Yttrium", symbol: "Y", mass: 88.906, electronegativity: 1.22, radius: 180.0, color: Color::srgb(0.6, 1.0, 1.0) },
    ElementData { name: "Zirconium", symbol: "Zr", mass: 91.224, electronegativity: 1.33, radius: 160.0, color: Color::srgb(0.6, 0.9, 0.9) },
    ElementData { name: "Niobium", symbol: "Nb", mass: 92.906, electronegativity: 1.60, radius: 146.0, color: Color::srgb(0.4, 0.8, 0.7) },
    ElementData { name: "Molybdenum", symbol: "Mo", mass: 95.950, electronegativity: 2.16, radius: 139.0, color: Color::srgb(0.3, 0.7, 0.6) },
    ElementData { name: "Technetium", symbol: "Tc", mass: 98.000, electronegativity: 1.90, radius: 136.0, color: Color::srgb(0.2, 0.5, 0.5) },
    ElementData { name: "Ruthenium", symbol: "Ru", mass: 101.070, electronegativity: 2.20, radius: 134.0, color: Color::srgb(0.1, 0.6, 0.6) },
    ElementData { name: "Rhodium", symbol: "Rh", mass: 102.906, electronegativity: 2.28, radius: 134.0, color: Color::srgb(0.0, 0.5, 0.5) },
    ElementData { name: "Palladium", symbol: "Pd", mass: 106.420, electronegativity: 2.20, radius: 137.0, color: Color::srgb(0.0, 0.4, 0.5) },
    ElementData { name: "Silver", symbol: "Ag", mass: 107.868, electronegativity: 1.93, radius: 144.0, color: Color::srgb(0.9, 0.9, 0.9) },
    ElementData { name: "Cadmium", symbol: "Cd", mass: 112.411, electronegativity: 1.69, radius: 151.0, color: Color::srgb(1.0, 0.8, 0.6) },
    ElementData { name: "Indium", symbol: "In", mass: 114.818, electronegativity: 1.78, radius: 167.0, color: Color::srgb(0.6, 0.5, 0.5) },
    ElementData { name: "Tin", symbol: "Sn", mass: 118.710, electronegativity: 1.96, radius: 140.0, color: Color::srgb(0.4, 0.5, 0.6) },
    ElementData { name: "Antimony", symbol: "Sb", mass: 121.760, electronegativity: 2.05, radius: 140.0, color: Color::srgb(0.6, 0.4, 0.8) },
    ElementData { name: "Tellurium", symbol: "Te", mass: 127.600, electronegativity: 2.10, radius: 142.0, color: Color::srgb(0.8, 0.4, 0.0) },
    ElementData { name: "Iodine", symbol: "I", mass: 126.904, electronegativity: 2.66, radius: 133.0, color: Color::srgb(0.4, 0.0, 0.4) },
    ElementData { name: "Xenon", symbol: "Xe", mass: 131.293, electronegativity: 2.60, radius: 130.0, color: Color::srgb(0.3, 0.6, 0.7) },

    // 55-71: Período 6 (Césio até Lutécio, incluindo Lantanídeos)
    ElementData { name: "Cesium", symbol: "Cs", mass: 132.905, electronegativity: 0.79, radius: 265.0, color: Color::srgb(0.3, 0.1, 0.5) },
    ElementData { name: "Barium", symbol: "Ba", mass: 137.327, electronegativity: 0.89, radius: 222.0, color: Color::srgb(0.0, 0.6, 0.0) },
    ElementData { name: "Lanthanum", symbol: "La", mass: 138.905, electronegativity: 1.10, radius: 187.0, color: Color::srgb(0.4, 0.8, 1.0) },
    ElementData { name: "Cerium", symbol: "Ce", mass: 140.116, electronegativity: 1.12, radius: 181.8, color: Color::srgb(1.0, 1.0, 0.8) },
    ElementData { name: "Praseodymium", symbol: "Pr", mass: 140.908, electronegativity: 1.13, radius: 182.4, color: Color::srgb(0.8, 1.0, 0.8) },
    ElementData { name: "Neodymium", symbol: "Nd", mass: 144.242, electronegativity: 1.14, radius: 181.4, color: Color::srgb(0.8, 1.0, 0.6) },
    ElementData { name: "Promethium", symbol: "Pm", mass: 145.000, electronegativity: 1.13, radius: 183.4, color: Color::srgb(0.6, 1.0, 0.6) },
    ElementData { name: "Samarium", symbol: "Sm", mass: 150.360, electronegativity: 1.17, radius: 180.4, color: Color::srgb(0.6, 1.0, 0.8) },
    ElementData { name: "Europium", symbol: "Eu", mass: 151.964, electronegativity: 1.20, radius: 208.4, color: Color::srgb(0.4, 1.0, 0.6) },
    ElementData { name: "Gadolinium", symbol: "Gd", mass: 157.250, electronegativity: 1.20, radius: 180.4, color: Color::srgb(0.3, 1.0, 0.8) },
    ElementData { name: "Terbium", symbol: "Tb", mass: 158.925, electronegativity: 1.20, radius: 177.3, color: Color::srgb(0.2, 1.0, 0.6) },
    ElementData { name: "Dysprosium", symbol: "Dy", mass: 162.500, electronegativity: 1.22, radius: 178.1, color: Color::srgb(0.1, 1.0, 0.8) },
    ElementData { name: "Holmium", symbol: "Ho", mass: 164.930, electronegativity: 1.23, radius: 176.2, color: Color::srgb(0.0, 1.0, 0.6) },
    ElementData { name: "Erbium", symbol: "Er", mass: 167.259, electronegativity: 1.24, radius: 176.1, color: Color::srgb(0.0, 0.9, 0.5) },
    ElementData { name: "Thulium", symbol: "Tm", mass: 168.934, electronegativity: 1.25, radius: 175.9, color: Color::srgb(0.0, 0.8, 0.4) },
    ElementData { name: "Ytterbium", symbol: "Yb", mass: 173.045, electronegativity: 1.10, radius: 192.9, color: Color::srgb(0.0, 0.7, 0.3) },
    ElementData { name: "Lutetium", symbol: "Lu", mass: 174.967, electronegativity: 1.27, radius: 173.4, color: Color::srgb(0.0, 0.6, 0.2) },
    
    // 72-86: Período 6 (Metais de Transição pesados até Radônio)
    ElementData { name: "Hafnium", symbol: "Hf", mass: 178.490, electronegativity: 1.30, radius: 159.0, color: Color::srgb(0.3, 0.7, 1.0) },
    ElementData { name: "Tantalum", symbol: "Ta", mass: 180.948, electronegativity: 1.50, radius: 146.0, color: Color::srgb(0.3, 0.6, 1.0) },
    ElementData { name: "Tungsten", symbol: "W", mass: 183.840, electronegativity: 2.36, radius: 139.0, color: Color::srgb(0.1, 0.5, 0.8) },
    ElementData { name: "Rhenium", symbol: "Re", mass: 186.207, electronegativity: 1.90, radius: 137.0, color: Color::srgb(0.1, 0.4, 0.6) },
    ElementData { name: "Osmium", symbol: "Os", mass: 190.230, electronegativity: 2.20, radius: 135.0, color: Color::srgb(0.1, 0.3, 0.5) },
    ElementData { name: "Iridium", symbol: "Ir", mass: 192.217, electronegativity: 2.20, radius: 135.5, color: Color::srgb(0.1, 0.2, 0.4) },
    ElementData { name: "Platinum", symbol: "Pt", mass: 195.084, electronegativity: 2.28, radius: 138.5, color: Color::srgb(0.8, 0.8, 0.9) },
    ElementData { name: "Gold", symbol: "Au", mass: 196.967, electronegativity: 2.54, radius: 144.0, color: Color::srgb(1.0, 0.8, 0.0) },
    ElementData { name: "Mercury", symbol: "Hg", mass: 200.592, electronegativity: 2.00, radius: 151.0, color: Color::srgb(0.7, 0.7, 0.8) },
    ElementData { name: "Thallium", symbol: "Tl", mass: 204.383, electronegativity: 1.62, radius: 170.0, color: Color::srgb(0.6, 0.3, 0.3) },
    ElementData { name: "Lead", symbol: "Pb", mass: 207.200, electronegativity: 2.33, radius: 175.0, color: Color::srgb(0.3, 0.3, 0.4) },
    ElementData { name: "Bismuth", symbol: "Bi", mass: 208.980, electronegativity: 2.02, radius: 154.0, color: Color::srgb(0.6, 0.3, 0.6) },
    ElementData { name: "Polonium", symbol: "Po", mass: 209.000, electronegativity: 2.00, radius: 168.0, color: Color::srgb(0.7, 0.3, 0.0) },
    ElementData { name: "Astatine", symbol: "At", mass: 210.000, electronegativity: 2.20, radius: 143.0, color: Color::srgb(0.5, 0.3, 0.2) },
    ElementData { name: "Radon", symbol: "Rn", mass: 222.000, electronegativity: 2.20, radius: 120.0, color: Color::srgb(0.3, 0.5, 0.5) },

    // 87-103: Período 7 (Frâncio até Laurêncio, incluindo Actinídeos)
    ElementData { name: "Francium", symbol: "Fr", mass: 223.000, electronegativity: 0.70, radius: 260.0, color: Color::srgb(0.3, 0.0, 0.3) },
    ElementData { name: "Radium", symbol: "Ra", mass: 226.000, electronegativity: 0.90, radius: 221.0, color: Color::srgb(0.0, 0.4, 0.0) },
    ElementData { name: "Actinium", symbol: "Ac", mass: 227.000, electronegativity: 1.10, radius: 195.0, color: Color::srgb(0.4, 0.6, 1.0) },
    ElementData { name: "Thorium", symbol: "Th", mass: 232.038, electronegativity: 1.30, radius: 179.0, color: Color::srgb(0.0, 0.8, 1.0) },
    ElementData { name: "Protactinium", symbol: "Pa", mass: 231.036, electronegativity: 1.50, radius: 163.0, color: Color::srgb(0.0, 0.6, 1.0) },
    ElementData { name: "Uranium", symbol: "U", mass: 238.029, electronegativity: 1.38, radius: 156.0, color: Color::srgb(0.2, 1.0, 0.2) },
    ElementData { name: "Neptunium", symbol: "Np", mass: 237.000, electronegativity: 1.36, radius: 155.0, color: Color::srgb(0.0, 0.4, 1.0) },
    ElementData { name: "Plutonium", symbol: "Pu", mass: 244.000, electronegativity: 1.28, radius: 159.0, color: Color::srgb(0.8, 0.0, 1.0) },
    ElementData { name: "Americium", symbol: "Am", mass: 243.000, electronegativity: 1.13, radius: 173.0, color: Color::srgb(0.3, 0.3, 0.8) },
    ElementData { name: "Curium", symbol: "Cm", mass: 247.000, electronegativity: 1.28, radius: 174.0, color: Color::srgb(0.5, 0.2, 0.7) },
    ElementData { name: "Berkelium", symbol: "Bk", mass: 247.000, electronegativity: 1.30, radius: 170.0, color: Color::srgb(0.6, 0.2, 0.6) },
    ElementData { name: "Californium", symbol: "Cf", mass: 251.000, electronegativity: 1.30, radius: 186.0, color: Color::srgb(0.7, 0.2, 0.5) },
    ElementData { name: "Einsteinium", symbol: "Es", mass: 252.000, electronegativity: 1.30, radius: 186.0, color: Color::srgb(0.8, 0.2, 0.4) },
    ElementData { name: "Fermium", symbol: "Fm", mass: 257.000, electronegativity: 1.30, radius: 190.0, color: Color::srgb(0.9, 0.2, 0.3) },
    ElementData { name: "Mendelevium", symbol: "Md", mass: 258.000, electronegativity: 1.30, radius: 190.0, color: Color::srgb(1.0, 0.3, 0.4) },
    ElementData { name: "Nobelium", symbol: "No", mass: 259.000, electronegativity: 1.30, radius: 190.0, color: Color::srgb(1.0, 0.4, 0.5) },
    ElementData { name: "Lawrencium", symbol: "Lr", mass: 262.000, electronegativity: 1.30, radius: 190.0, color: Color::srgb(0.9, 0.0, 0.4) },

    // 104-118: Transactanídeos (Superpesados, valores em grande parte teóricos)
    ElementData { name: "Rutherfordium", symbol: "Rf", mass: 267.000, electronegativity: 0.00, radius: 150.0, color: Color::srgb(0.8, 0.0, 0.2) },
    ElementData { name: "Dubnium", symbol: "Db", mass: 270.000, electronegativity: 0.00, radius: 150.0, color: Color::srgb(0.7, 0.0, 0.2) },
    ElementData { name: "Seaborgium", symbol: "Sg", mass: 271.000, electronegativity: 0.00, radius: 150.0, color: Color::srgb(0.6, 0.0, 0.2) },
    ElementData { name: "Bohrium", symbol: "Bh", mass: 270.000, electronegativity: 0.00, radius: 150.0, color: Color::srgb(0.5, 0.0, 0.2) },
    ElementData { name: "Hassium", symbol: "Hs", mass: 277.000, electronegativity: 0.00, radius: 150.0, color: Color::srgb(0.4, 0.0, 0.2) },
    ElementData { name: "Meitnerium", symbol: "Mt", mass: 276.000, electronegativity: 0.00, radius: 150.0, color: Color::srgb(0.3, 0.0, 0.2) },
    ElementData { name: "Darmstadtium", symbol: "Ds", mass: 281.000, electronegativity: 0.00, radius: 150.0, color: Color::srgb(0.2, 0.0, 0.2) },
    ElementData { name: "Roentgenium", symbol: "Rg", mass: 280.000, electronegativity: 0.00, radius: 150.0, color: Color::srgb(0.1, 0.0, 0.2) },
    ElementData { name: "Copernicium", symbol: "Cn", mass: 285.000, electronegativity: 0.00, radius: 150.0, color: Color::srgb(0.0, 0.0, 0.2) },
    ElementData { name: "Nihonium", symbol: "Nh", mass: 284.000, electronegativity: 0.00, radius: 150.0, color: Color::srgb(0.0, 0.2, 0.4) },
    ElementData { name: "Flerovium", symbol: "Fl", mass: 289.000, electronegativity: 0.00, radius: 150.0, color: Color::srgb(0.0, 0.4, 0.6) },
    ElementData { name: "Moscovium", symbol: "Mc", mass: 288.000, electronegativity: 0.00, radius: 150.0, color: Color::srgb(0.0, 0.6, 0.8) },
    ElementData { name: "Livermorium", symbol: "Lv", mass: 293.000, electronegativity: 0.00, radius: 150.0, color: Color::srgb(0.0, 0.8, 1.0) },
    ElementData { name: "Tennessine", symbol: "Ts", mass: 294.000, electronegativity: 0.00, radius: 150.0, color: Color::srgb(0.0, 1.0, 0.8) },
    ElementData { name: "Oganesson", symbol: "Og", mass: 294.000, electronegativity: 0.00, radius: 157.0, color: Color::srgb(1.0, 0.0, 0.5) },
];

pub const METAL_ELEMENTS: &[&str] = &[
    "Lithium", "Sodium", "Potassium", "Rubidium", "Cesium", "Francium",
    "Beryllium", "Magnesium", "Calcium", "Strontium", "Barium", "Radium",
    "Scandium", "Titanium", "Vanadium", "Chromium", "Manganese", "Iron", "Cobalt", "Nickel", "Copper", "Zinc",
    "Yttrium", "Zirconium", "Niobium", "Molybdenum", "Technetium", "Ruthenium", "Rhodium", "Palladium", "Silver", "Cadmium",
    "Hafnium", "Tantalum", "Tungsten", "Rhenium", "Osmium", "Iridium", "Platinum", "Gold", "Mercury",
    "Rutherfordium", "Dubnium", "Seaborgium", "Bohrium", "Hassium", "Meitnerium", "Darmstadtium", "Roentgenium", "Copernicium",
    "Aluminum", "Gallium", "Indium", "Tin", "Thallium", "Lead", "Bismuth", "Nihonium", "Flerovium", "Moscovium", "Livermorium",
    "Lanthanum", "Cerium", "Praseodymium", "Neodymium", "Promethium", "Samarium", "Europium", "Gadolinium", "Terbium", "Dysprosium", "Holmium", "Erbium", "Thulium", "Ytterbium", "Lutetium",
    "Actinium", "Thorium", "Protactinium", "Uranium", "Neptunium", "Plutonium", "Americium", "Curium", "Berkelium", "Californium", "Einsteinium", "Fermium", "Mendelevium", "Nobelium", "Lawrencium"
];

pub const NON_METAL_ELEMENTS: &[&str] = &[
    "Hydrogen", "Carbon", "Nitrogen", "Oxygen", "Phosphorus", "Sulfur", "Selenium"
];

pub const NOBLE_GASES: &[&str] = &[
    "Helium", "Neon", "Argon", "Krypton", "Xenon", "Radon", "Oganesson"
];

pub const HALOGENS: &[&str] = &[
    "Fluorine", "Chlorine", "Bromine", "Iodine", "Astatine", "Tennessine"
];

pub const METALLOIDS: &[&str] = &[
    "Boron", "Silicon", "Germanium", "Arsenic", "Antimony", "Tellurium", "Polonium"
];

pub const PERIODIC_TABLE_SIZE: usize = PERIODIC_TABLE.len();

pub fn get_element_by_name(name: &str) -> &'static ElementData {
    PERIODIC_TABLE.iter().find(|e| e.name == name).unwrap_or(&PERIODIC_TABLE[0])
}

pub fn get_elements_by_category(category: &[&str]) -> Vec<&'static ElementData> {
    PERIODIC_TABLE
        .iter()
        .filter(|e| category.contains(&e.name))
        .collect()
}