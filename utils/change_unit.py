def fahrenheit_to_celsius(T_fahrenheit):
    T_celsius = (T_fahrenheit - 32) * 5 / 9
    return T_celsius


def inchesH2O_to_Pascal(pressure_inchesH2O):
    pressure_Pascal = pressure_inchesH2O * 249.0889
    return pressure_Pascal


def CFM_to_m3h(air_flow_CFM):
    air_flow_m3h = air_flow_CFM * 1.6990107955
    return air_flow_m3h
