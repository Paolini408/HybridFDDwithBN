import bnlearn as bn
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD
from utils.change_unit import fahrenheit_to_celsius, inchesH2O_to_Pascal


def obtain_models_mech():
    model_tot = BayesianNetwork([('Cooling Coil', 'supply'),
                                 ('Cooling Coil', 'AHU'), ('Cooling Coil', 'CHWC_VLV_DM'),
                                 ('Economizer', 'AHU'), ('Economizer', 'MA_TEMP'),
                                 ('SAT_Sensor', 'AHU'), ('SAT_Sensor', 'Rule_BIAS'),
                                 ('MAT_Sensor', 'AHU'), ('MAT_Sensor', 'Rule_MAT'),
                                 ('Air duct', 'AHU'), ('Air duct', 'SF_power'), ('Air duct', 'RF_power'), ('Air duct', 'pressure')])

    model_CC = BayesianNetwork([('Cooling Coil', 'supply'), ('Cooling Coil', 'CHWC_VLV_DM')])

    model_ECO = BayesianNetwork([('Economizer', 'MA_TEMP')])

    model_SENSOR = BayesianNetwork([('SAT_Sensor', 'Rule_BIAS')])

    model_MAT = BayesianNetwork([('MAT_Sensor', 'Rule_MAT')])

    model_DUCT = BayesianNetwork([('Air duct', 'SF_power'), ('Air duct', 'RF_power'), ('Air duct', 'pressure')])

    cpt_AHU_mech = TabularCPD(variable='AHU', variable_card=2, values=[[1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                                                        1, 0],
                                                                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                                        0, 1]],
                              evidence=['Cooling Coil', 'Economizer', 'SAT_Sensor', 'MAT_Sensor', 'Air duct'],
                              evidence_card=[2, 2, 2, 2, 2])

    cpt_CC_mech = TabularCPD(variable='Cooling Coil', variable_card=2, values=[[0.1], [0.9]])

    cpt_ECO_mech = TabularCPD(variable='Economizer', variable_card=2, values=[[0.1], [0.9]])

    cpt_SENSOR_mech = TabularCPD(variable='SAT_Sensor', variable_card=2, values=[[0.1], [0.9]])

    cpt_MAT_mech = TabularCPD(variable='MAT_Sensor', variable_card=2, values=[[0.1], [0.9]])

    cpt_DUCT_mech = TabularCPD(variable='Air duct', variable_card=2, values=[[0.1], [0.9]])

    cpt_supply_mech = TabularCPD(variable='supply', variable_card=2, values=[[0.99, 0.01], [0.01, 0.99]],
                                 evidence=['Cooling Coil'], evidence_card=[2])

    cpt_pressure_mech = TabularCPD(variable='pressure', variable_card=2, values=[[0.9, 0.1], [0.1, 0.9]],
                                   evidence=['Air duct'], evidence_card=[2])

    cpt_MA_mech = TabularCPD(variable='MA_TEMP', variable_card=2, values=[[0.995, 0.005], [0.005, 0.995]],
                             evidence=['Economizer'], evidence_card=[2])

    cpt_rule_bias = TabularCPD(variable='Rule_BIAS', variable_card=2, values=[[0.99, 0.01], [0.01, 0.99]],
                               evidence=['SAT_Sensor'], evidence_card=[2])

    cpt_ccv = TabularCPD(variable='CHWC_VLV_DM', variable_card=2, values=[[0.95, 0.05], [0.05, 0.95]],
                         evidence=['Cooling Coil'], evidence_card=[2])

    cpt_rule_mat = TabularCPD(variable='Rule_MAT', variable_card=2, values=[[0.99, 0.01], [0.01, 0.99]],
                              evidence=['MAT_Sensor'], evidence_card=[2])

    cpt_sf_power = TabularCPD(variable='SF_power', variable_card=2, values=[[0.9, 0.1], [0.1, 0.9]],
                              evidence=['Air duct'], evidence_card=[2])

    cpt_rf_power = TabularCPD(variable='RF_power', variable_card=2, values=[[0.9, 0.1], [0.1, 0.9]],
                              evidence=['Air duct'], evidence_card=[2])

    model_tot.add_cpds(cpt_AHU_mech, cpt_CC_mech, cpt_ECO_mech, cpt_SENSOR_mech, cpt_supply_mech, cpt_MA_mech,
                       cpt_rule_bias, cpt_ccv, cpt_MAT_mech, cpt_DUCT_mech, cpt_rule_mat, cpt_sf_power, cpt_rf_power, cpt_pressure_mech)
    model_CC.add_cpds(cpt_CC_mech, cpt_supply_mech, cpt_ccv)
    model_ECO.add_cpds(cpt_ECO_mech, cpt_MA_mech)
    model_SENSOR.add_cpds(cpt_SENSOR_mech, cpt_rule_bias)
    model_MAT.add_cpds(cpt_MAT_mech, cpt_rule_mat)
    model_DUCT.add_cpds(cpt_DUCT_mech, cpt_sf_power, cpt_rf_power, cpt_pressure_mech)

    return model_tot, model_CC, model_ECO, model_SENSOR, model_MAT, model_DUCT


def obtain_models_eco():
    model_tot = BayesianNetwork([('Economizer', 'mixed'),
                                 ('Cooling Coil', 'AHU'), ('Cooling Coil', 'SA_TEMP'),
                                 ('Economizer', 'AHU'), ('Economizer', 'OA_DMPR_DM'),
                                 ('SAT_Sensor', 'AHU'), ('SAT_Sensor', 'Rule_BIAS'),
                                 ('MAT_Sensor', 'AHU'), ('MAT_Sensor', 'Rule_MAT'),
                                 ('Air duct', 'AHU'), ('Air duct', 'SF_power'), ('Air duct', 'RF_power'), ('Air duct', 'pressure')])

    model_CC = BayesianNetwork([('Cooling Coil', 'SA_TEMP')])

    model_ECO = BayesianNetwork([('Economizer', 'OA_DMPR_DM'), ('Economizer', 'mixed')])

    model_SENSOR = BayesianNetwork([('SAT_Sensor', 'Rule_BIAS')])

    model_MAT = BayesianNetwork([('MAT_Sensor', 'Rule_MAT')])

    model_DUCT = BayesianNetwork([('Air duct', 'SF_power'), ('Air duct', 'RF_power'), ('Air duct', 'pressure')])

    cpt_AHU_eco = TabularCPD(variable='AHU', variable_card=2, values=[[1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                                                       1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                                                       1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                                                       1, 0],
                                                                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                                       0, 1]],
                             evidence=['Cooling Coil', 'Economizer', 'SAT_Sensor', 'MAT_Sensor', 'Air duct'],
                             evidence_card=[2, 2, 2, 2, 2])

    cpt_CC_eco = TabularCPD(variable='Cooling Coil', variable_card=2, values=[[0.1], [0.9]])

    cpt_ECO_eco = TabularCPD(variable='Economizer', variable_card=2, values=[[0.1], [0.9]])

    cpt_SENSOR_eco = TabularCPD(variable='SAT_Sensor', variable_card=2, values=[[0.1], [0.9]])

    cpt_MAT_eco = TabularCPD(variable='MAT_Sensor', variable_card=2, values=[[0.1], [0.9]])

    cpt_DUCT_eco = TabularCPD(variable='Air duct', variable_card=2, values=[[0.1], [0.9]])

    cpt_mixed_eco = TabularCPD(variable='mixed', variable_card=2, values=[[0.995, 0.005], [0.005, 0.995]],
                               evidence=['Economizer'], evidence_card=[2])

    cpt_pressure_eco = TabularCPD(variable='pressure', variable_card=2, values=[[0.9, 0.1], [0.1, 0.9]],
                                  evidence=['Air duct'], evidence_card=[2])

    cpt_rule_bias = TabularCPD(variable='Rule_BIAS', variable_card=2, values=[[0.99, 0.01], [0.01, 0.99]],
                               evidence=['SAT_Sensor'], evidence_card=[2])

    cpt_OA_damper = TabularCPD(variable='OA_DMPR_DM', variable_card=2, values=[[0.95, 0.05], [0.05, 0.95]],
                               evidence=['Economizer'], evidence_card=[2])

    cpt_SA_eco = TabularCPD(variable='SA_TEMP', variable_card=2, values=[[0.95, 0.05], [0.05, 0.95]],
                            evidence=['Cooling Coil'], evidence_card=[2])

    cpt_rule_mat = TabularCPD(variable='Rule_MAT', variable_card=2, values=[[0.99, 0.01], [0.01, 0.99]],
                              evidence=['MAT_Sensor'], evidence_card=[2])

    cpt_sf_power = TabularCPD(variable='SF_power', variable_card=2, values=[[0.9, 0.1], [0.1, 0.9]],
                              evidence=['Air duct'], evidence_card=[2])

    cpt_rf_power = TabularCPD(variable='RF_power', variable_card=2, values=[[0.9, 0.1], [0.1, 0.9]],
                              evidence=['Air duct'], evidence_card=[2])

    model_tot.add_cpds(cpt_AHU_eco, cpt_CC_eco, cpt_ECO_eco, cpt_SENSOR_eco, cpt_mixed_eco, cpt_SA_eco,
                       cpt_rule_bias, cpt_OA_damper, cpt_MAT_eco, cpt_DUCT_eco, cpt_rule_mat, cpt_sf_power, cpt_rf_power, cpt_pressure_eco)
    model_CC.add_cpds(cpt_CC_eco, cpt_SA_eco)
    model_ECO.add_cpds(cpt_ECO_eco, cpt_OA_damper, cpt_mixed_eco)
    model_SENSOR.add_cpds(cpt_SENSOR_eco, cpt_rule_bias)
    model_MAT.add_cpds(cpt_MAT_eco, cpt_rule_mat)
    model_DUCT.add_cpds(cpt_DUCT_eco, cpt_sf_power, cpt_rf_power, cpt_pressure_eco)

    return model_tot, model_CC, model_ECO, model_SENSOR, model_MAT, model_DUCT


def obtain_evidence_mech(row, name_list, rmse_list):
    virtual_evidence_TOT = [
        TabularCPD('CHWC_VLV_DM', 2, [[row['p_fault_' + name_list[3]]], [(1 - row['p_fault_' + name_list[3]])]]),
        TabularCPD('MA_TEMP', 2, [[row['p_fault_' + name_list[0]]], [1 - row['p_fault_' + name_list[0]]]]),
        TabularCPD('SF_power', 2, [[row['p_fault_' + name_list[4]]], [1 - row['p_fault_' + name_list[4]]]]),
        TabularCPD('RF_power', 2, [[row['p_fault_' + name_list[5]]], [1 - row['p_fault_' + name_list[5]]]])
    ]

    virtual_evidence_CC = [
        TabularCPD('CHWC_VLV_DM', 2, [[row['p_fault_' + name_list[3]]], [(1 - row['p_fault_' + name_list[3]])]])
    ]

    virtual_evidence_ECO = [
        TabularCPD('MA_TEMP', 2, [[row['p_fault_' + name_list[0]]], [1 - row['p_fault_' + name_list[0]]]])
    ]

    virtual_evidence_DUCT = [
        TabularCPD('SF_power', 2, [[row['p_fault_' + name_list[4]]], [1 - row['p_fault_' + name_list[4]]]]),
        TabularCPD('RF_power', 2, [[row['p_fault_' + name_list[5]]], [1 - row['p_fault_' + name_list[5]]]])
    ]

    hard_evidence_TOT = {}
    hard_evidence_CC = {}
    hard_evidence_SENSOR = {}
    hard_evidence_MAT = {}
    hard_evidence_pressure = {}
    if abs(row['SA_TEMP'] - fahrenheit_to_celsius(55)) > 0.25:
        hard_evidence_TOT['supply'] = 0  # Label to mean 'Fault'
        hard_evidence_CC['supply'] = 0
    else:
        hard_evidence_TOT['supply'] = 1  # Label to mean 'Normal'
        hard_evidence_CC['supply'] = 1

    if abs(row['SA_TEMP'] - fahrenheit_to_celsius(55)) < 0.25 and abs(row['dev_CHWC_VLV_DM']) > rmse_list[3] * 3:
        hard_evidence_TOT['Rule_BIAS'] = 0
        hard_evidence_SENSOR['Rule_BIAS'] = 0
    else:
        hard_evidence_TOT['Rule_BIAS'] = 1
        hard_evidence_SENSOR['Rule_BIAS'] = 1

    if abs(row['dev_MA_TEMP']) > 1.75:
        hard_evidence_TOT['Rule_MAT'] = 0
        hard_evidence_MAT['Rule_MAT'] = 0
    else:
        hard_evidence_TOT['Rule_MAT'] = 1
        hard_evidence_MAT['Rule_MAT'] = 1

    if abs(row['SA_SP'] - inchesH2O_to_Pascal(1.6)) > 15:
        hard_evidence_TOT['pressure'] = 0
        hard_evidence_pressure['pressure'] = 0
    else:
        hard_evidence_TOT['pressure'] = 1
        hard_evidence_pressure['pressure'] = 1

    return (virtual_evidence_TOT, virtual_evidence_CC, virtual_evidence_ECO, virtual_evidence_DUCT,
            hard_evidence_TOT, hard_evidence_CC, hard_evidence_SENSOR, hard_evidence_MAT, hard_evidence_pressure)


def obtain_evidence_eco(row, name_list, rmse_list):
    virtual_evidence_TOT = [
        TabularCPD('SA_TEMP', 2, [[row['p_fault_' + name_list[1]]], [1 - row['p_fault_' + name_list[1]]]]),
        TabularCPD('OA_DMPR_DM', 2, [[row['p_fault_' + name_list[2]]], [1 - row['p_fault_' + name_list[2]]]]),
        TabularCPD('SF_power', 2, [[row['p_fault_' + name_list[4]]], [1 - row['p_fault_' + name_list[4]]]]),
        TabularCPD('RF_power', 2, [[row['p_fault_' + name_list[5]]], [1 - row['p_fault_' + name_list[5]]]])
    ]

    virtual_evidence_CC = [
        TabularCPD('SA_TEMP', 2, [[row['p_fault_' + name_list[1]]], [1 - row['p_fault_' + name_list[1]]]])
    ]

    virtual_evidence_ECO = [
        TabularCPD('OA_DMPR_DM', 2, [[row['p_fault_' + name_list[2]]], [1 - row['p_fault_' + name_list[2]]]])
    ]

    virtual_evidence_DUCT = [
        TabularCPD('SF_power', 2, [[row['p_fault_' + name_list[4]]], [1 - row['p_fault_' + name_list[4]]]]),
        TabularCPD('RF_power', 2, [[row['p_fault_' + name_list[5]]], [1 - row['p_fault_' + name_list[5]]]])
    ]

    hard_evidence_TOT = {}
    hard_evidence_ECO = {}
    hard_evidence_SENSOR = {}
    hard_evidence_MAT = {}
    hard_evidence_pressure = {}
    if abs(row['MA_TEMP'] - fahrenheit_to_celsius(55)) > 0.25:
        hard_evidence_TOT['mixed'] = 0  # Fault
        hard_evidence_ECO['mixed'] = 0
    else:
        hard_evidence_TOT['mixed'] = 1  # Normal
        hard_evidence_ECO['mixed'] = 1

    if abs(row['SA_TEMP'] - row['MA_TEMP']) > 1.75:
        hard_evidence_TOT['Rule_BIAS'] = 0  # Fault
        hard_evidence_SENSOR['Rule_BIAS'] = 0
    else:
        hard_evidence_TOT['Rule_BIAS'] = 1  # Normal
        hard_evidence_SENSOR['Rule_BIAS'] = 1

    if abs(row['MA_TEMP'] - fahrenheit_to_celsius(55)) < 0.25 and abs(row['dev_MA_TEMP']) > rmse_list[0] * 3:
        hard_evidence_TOT['Rule_MAT'] = 0
        hard_evidence_MAT['Rule_MAT'] = 0
    else:
        hard_evidence_TOT['Rule_MAT'] = 1
        hard_evidence_MAT['Rule_MAT'] = 1

    if abs(row['SA_SP'] - inchesH2O_to_Pascal(1.6)) > 15:
        hard_evidence_TOT['pressure'] = 0
        hard_evidence_pressure['pressure'] = 0
    else:
        hard_evidence_TOT['pressure'] = 1
        hard_evidence_pressure['pressure'] = 1

    return (virtual_evidence_TOT, virtual_evidence_CC, virtual_evidence_ECO, virtual_evidence_DUCT,
            hard_evidence_TOT, hard_evidence_ECO, hard_evidence_SENSOR, hard_evidence_MAT, hard_evidence_pressure)
