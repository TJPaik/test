
REGRESSION_POWER_NETS = ["VDD"]
REGRESSION_GROUND_NETS = ["0"]
REGRESSION_INSTANCE_TYPES = ['mos', 'capacitor', 'inductor', 'resistor', 'vsource', 'isource']

REGRESSION_ALIAS_TO_CANON_DICT = {
    'c': 'value', # capacitor
    'r': 'value', # resistor
    'l': 'value', # inductor
    'dc': 'value', # vsource
}
REGRESSION_NODE_ATTRIBUTES = ['w', 'l', 'as', 'ad', 'ps', 'pd', 'm', 'value', 'mag', 'phase', 'value']

TARGET_LIST = [ 
    "Bandwidth", # CSVA, CVA, LNA, Transmitter, TSVA 
    "PowerConsumption", # CSVA, CVA, LNA, PA, Receiver, Transmitter, TSVA, VCO 
    "VoltageGain", # CSVA, CVA, Receiver, TSVA 
    "IFVoltageSwing", # Mixer 
    "ConversionGain", # Mixer 
    "NoiseFigure", # Mixer, Receiver 
    
    "LargeSignalPowerGain", # PA 
    "PAE", # PA 
    "DrainEfficiency", # PA 
    "PSAT", # PA 
    
    "PowerGain_LNA", # Receiver 
    "NoiseFigure_LNA", # Receiver 
    "S11_LNA", # Receiver 
    "ConversionGain_Mixer", # Receiver 
    "VoltageSwing_Mixer", # Receiver 
    "VoltageGain_Amp", # Receiver 
    
    "OutputPower", # Transmitter, VCO 
    "VoltageSwing", # Transmitter 
    "TuningRange_VCO", # Transmitter 
    "PhaseNoise_VCO", # Transmitter 
    "LargeSignal_PowerGain_PA", # Transmitter 
    "DrainEfficiency_PA", # Transmitter 
    "PAE_PA", # Transmitter 
    
    "OscillationFrequency", # VCO 
    "PhaseNoise", # VCO 
    "TuningRange" # VCO 
]
ALIAS_TO_CANON = { 
    # VCO 
    "TuningRange_VCO": "TuningRange", 
    "PhaseNoise_VCO": "PhaseNoise", 
    # PA (and PA inside Transmitter) 
    #"LargeSignal_PowerGain_PA": "LargeSignalPowerGain", 
    #"LargeSignalPowerGain": "LargeSignalPowerGain", 
    "LargeSignal_PowerGain_PA": "PowerGain", 
    "LargeSignalPowerGain": "PowerGain", 
    "PowerGain_LNA": "PowerGain",
    # idempotent 
    "DrainEfficiency_PA": "DrainEfficiency",
    "PAE_PA": "PAE", 
    # Mixer / Receiver 
    "ConversionGain_Mixer": "ConversionGain", 
    "IFVoltageSwing": "VoltageSwing", 
    "VoltageSwing_Mixer": "VoltageSwing", 
    "VoltageGain_Amp": "VoltageGain", 
    "NoiseFigure_LNA": "NoiseFigure", 
    "S11_LNA": "S11" 
}


CANON_TARGETS = [
    "Bandwidth", "PowerConsumption", "VoltageGain", "PowerGain", "ConversionGain",
    "NoiseFigure", "VoltageSwing", "OutputPower",
    #"LargeSignalPowerGain", 
    "PAE", "DrainEfficiency", "PSAT",
    "S11", "OscillationFrequency", "PhaseNoise", "TuningRange",
]

