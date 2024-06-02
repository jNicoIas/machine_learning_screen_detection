#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 12:17:10 2024

@author: jnicolas
"""

from machine_learning_module import SensordataProcessor

# if __name__ == "__main__": 

    # distance: 40 cm
    gl5528_40 = SensordataProcessor('S2-3200-GL5528-40CM.csv')
    gl5528_40.export_prints_to_csv('results_S2-3200-GL5528-40CM.csv')
    
    temt_40 = SensordataProcessor('S2-3200-TEMT-40CM.csv')
    temt_40.export_prints_to_csv('results_S2-3200-TEMT-40CM.csv')
    veml_40 = SensordataProcessor('S2-3200-VEML-40cm.csv')
    
    veml_40.export_prints_to_csv('results_S2-3200-VEML-40cm.csv')
    
    gl3472_40 = SensordataProcessor('S3-3472-40CM.csv')
    gl3472_40.export_prints_to_csv('results_S3-3472-40CM.csv')
    
    
    # distance: 20 cm
    gl5528_20 = SensordataProcessor('S2-3200-GL5528-20CM.csv')
    gl5528_20.export_prints_to_csv('results_S2-3200-GL5528-20CM.csv')
    
    # distance: 30 cm
    
    # distance: 50 cm
    
    