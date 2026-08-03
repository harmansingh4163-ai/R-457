#!/usr/bin/env python3
"""
build_kb.py — build the R-457 knowledge bank for the SD card.

Produces a binary fact store that an ESP32 can binary-search directly from SD
without loading it into RAM. Facts are short declarative sentences in exactly
the shape R-457 reasons over ("Facts: ...").

WHY THIS FORMAT
SD over SPI is slow (a few MB/s) and the ESP32 has no room to hold an index in
RAM for a large bank. So: a fixed-width sorted key index (binary search, about
log2(N) seeks) plus a variable-length record area.

  kb.bin layout
    magic     8 bytes   "R457KB01"
    n_keys    u32
    idx_off   u32       byte offset of the index
    records            key_len u8 | key | n_facts u8 | (len u16 | text)*
    index             n_keys * 32 bytes: key[26] (lowercase, NUL-padded,
                                          truncated) + rec_off u32 + rec_len u16
                      sorted bytewise by key -> binary search on device

SOURCES
  Built in here (authored, public-domain data — physical constants and
  formulas are facts, not copyrightable expression):
    * NIST/SI physical constants        * SI prefixes and unit conversions
    * EE formulas (Ohm, power, R/C/L)   * E24 resistor series, colour codes
    * Wire gauge ampacity               * Elements 1-20
    * Planets and solar system          * Common engineering rules of thumb

  Optional, downloaded by YOU on the Mac (this sandbox has no network for
  them). Each has a converter below; pass the file and it is merged:
    --wordnet   WordNet index.sense / data.noun  (princeton, permissive)
    --factbook  factbook.json (github.com/factbook/factbook.json, US gov PD)
    --csv       any "key,fact" CSV you write yourself

USAGE
  python3 build_kb.py --out kb.bin
  python3 build_kb.py --out kb.bin --factbook factbook.json --wordnet data.noun
  python3 build_kb.py --out kb.bin --query "copper"        # test a lookup
"""

import argparse, json, os, struct, sys
from collections import defaultdict

KEY_BYTES = 26
IDX_ENTRY = KEY_BYTES + 4 + 2          # 32 bytes

# ===========================================================================
# AUTHORED CORE  — (key, fact sentence)
# ===========================================================================


# quantities that should point at the formulas mentioning them
FORMULA_ALIASES = {
    "voltage": ["ohms law","electrical power","rms voltage","transformer ratio"],
    "current": ["ohms law","electrical power","charge","kirchhoff current"],
    "resistance": ["ohms law","series resistance","parallel resistance",
                   "wire resistance","power resistance"],
    "power": ["electrical power","power resistance","mechanical power"],
    "capacitance": ["series capacitance","parallel capacitance",
                    "capacitive reactance","rc time constant"],
    "inductance": ["inductive reactance","rl time constant","inductor energy"],
    "frequency": ["capacitive reactance","inductive reactance",
                  "resonant frequency","wave speed"],
    "force": ["newtons second law","work","pressure"],
    "energy": ["kinetic energy","potential energy","capacitor energy"],
    "velocity": ["momentum","kinetic energy","velocity"],
    "speed": ["velocity","wave speed"],
    "acceleration": ["newtons second law","acceleration"],
    "mass": ["newtons second law","momentum","density","kinetic energy"],
    "density": ["density"],
    "pressure": ["pressure"],
    "work": ["work","mechanical power"],
    "charge": ["charge"],
    "momentum": ["momentum"],
    # --- quantities introduced by extended_facts() ---
    "gain": ["inverting amplifier","noninverting amplifier","decibel","bjt gain"],
    "torque": ["torque"],
    "efficiency": ["efficiency"],
    "stress": ["stress","youngs modulus"],
    "strain": ["strain","youngs modulus"],
    "friction": ["friction"],
    "heat": ["specific heat formula","thermal conduction","latent heat fusion"],
    "temperature": ["ideal gas law","specific heat formula","thermal expansion"],
    "duty": ["duty cycle","pwm average"],
    "period": ["frequency period","pendulum period"],
    "runtime": ["battery runtime","battery capacity"],
    "impedance": ["capacitive reactance","inductive reactance"],
    "reactance": ["capacitive reactance","inductive reactance"],
    "divider": ["voltage divider","current divider"],
    "led": ["led resistor","red led","green led","blue led"],
    "resistor": ["led resistor","pull up resistor","series resistance",
                 "parallel resistance"],
    "diode": ["silicon diode","schottky diode","zener diode"],
    "flow": ["thermal conduction"],
    "moles": ["ideal gas law"],
}

def formula_aliases(facts):
    """Point quantity words at the formulas that use them, so 'what is the
    voltage' retrieves Ohm's law."""
    by_key = {}
    for k, t in facts:
        by_key.setdefault(k, []).append(t)
    out = []
    for q, keys in FORMULA_ALIASES.items():
        for k in keys:
            for t in by_key.get(k, []):
                out.append((q, t))
    return out


# ===========================================================================
#  ADD YOUR OWN FACTS HERE
#  ---------------------------------------------------------------------
#  Each line is:      ("key", "The fact written as a sentence."),
#
#    key   - lowercase words someone would type when asking. This is what the
#            search looks for. Two words is usually better than one.
#    fact  - ONE plain sentence. The model reads it as a given fact, so write
#            it the way you would want it repeated back.
#
#  Two shapes work best, because they are what the model was trained on:
#    a FORMULA it can put numbers into:
#        ("boost converter",
#         "Boost converter output voltage equals input voltage divided by 1 minus the duty cycle."),
#    a VALUE it can quote:
#        ("thermistor",
#         "An NTC thermistor decreases in resistance as temperature rises."),
#
#  Rules that will bite you if broken:
#    - every line ends with a COMMA
#    - both parts are in "double quotes"
#    - keep the indentation as it is below
#    - do not use a double quote inside the sentence
# ===========================================================================
def my_facts():
    return [

        # ---- delete these two examples and write your own ----
        ("boost converter",
         "Boost converter output voltage equals input voltage divided by 1 minus the duty cycle."),
        ("thermistor",
         "An NTC thermistor decreases in resistance as temperature rises."),

    ]
# ===========================================================================


def extended_facts():
    """Second wave of authored facts. Everything here is either a FORMULA the
    model can substitute values into, or a LOOKUP VALUE it can cite — the two
    shapes it was actually trained to handle. Prose explanations are omitted on
    purpose; the model reasons over facts, it does not paraphrase essays."""
    F = []

    # ---- op-amp configurations -------------------------------------------
    F += [
        ("inverting amplifier",
         "For an inverting amplifier, gain equals minus the feedback resistance divided by the input resistance."),
        ("noninverting amplifier",
         "For a non-inverting amplifier, gain equals 1 plus the feedback resistance divided by the ground resistance."),
        ("voltage follower",
         "A voltage follower has a gain of 1 and a very high input impedance."),
        ("op amp rules",
         "With negative feedback an ideal op amp draws no input current and holds its two inputs at the same voltage."),
        ("summing amplifier",
         "A summing amplifier output is minus the feedback resistance times the sum of each input voltage divided by its own resistor."),
        ("comparator",
         "A comparator output goes high when the non-inverting input is above the inverting input."),
    ]

    # ---- more circuit formulas -------------------------------------------
    F += [
        ("voltage divider",
         "Output voltage equals input voltage times the lower resistance divided by the sum of both resistances."),
        ("current divider",
         "In two parallel branches, the current in one branch equals total current times the opposite resistance divided by the sum."),
        ("thevenin",
         "Any linear network can be replaced by one voltage source in series with one resistance."),
        ("norton",
         "Any linear network can be replaced by one current source in parallel with one resistance."),
        ("led resistor",
         "The LED series resistance equals supply voltage minus LED forward voltage, divided by the desired current."),
        ("power factor",
         "Power factor equals real power divided by apparent power."),
        ("apparent power",
         "Apparent power equals RMS voltage times RMS current."),
        ("q factor",
         "The Q factor equals the resonant frequency divided by the bandwidth."),
        ("duty cycle",
         "Duty cycle equals the on time divided by the period."),
        ("pwm average",
         "The average PWM voltage equals the duty cycle times the supply voltage."),
        ("frequency period",
         "Frequency equals 1 divided by the period."),
        ("angular frequency",
         "Angular frequency equals 2 pi times the frequency."),
        ("peak to peak",
         "For a sine wave, peak to peak voltage is twice the peak voltage."),
        ("rms current",
         "For a sine wave, RMS current is the peak current divided by the square root of 2."),
        ("capacitor charge",
         "The charge on a capacitor equals capacitance times voltage."),
        ("coulombs law",
         "The force between two charges equals k times the product of the charges divided by the distance squared."),
        ("magnetic force",
         "The force on a current-carrying wire equals magnetic flux density times current times length."),
        ("transformer power",
         "In an ideal transformer the input power equals the output power."),
    ]

    # ---- semiconductors ---------------------------------------------------
    F += [
        ("silicon diode",   "A silicon diode has a forward voltage drop of about 0.7 volts."),
        ("schottky diode",  "A Schottky diode has a forward voltage drop of about 0.3 volts."),
        ("red led",         "A red LED has a forward voltage of about 1.8 volts."),
        ("green led",       "A green LED has a forward voltage of about 2.2 volts."),
        ("blue led",        "A blue LED has a forward voltage of about 3.2 volts."),
        ("white led",       "A white LED has a forward voltage of about 3.2 volts."),
        ("zener diode",     "A Zener diode holds a steady voltage across itself when reverse biased past its breakdown voltage."),
        ("bjt gain",        "In a bipolar transistor, collector current equals beta times base current."),
        ("mosfet gate",     "A MOSFET is voltage controlled and its gate draws almost no steady current."),
        ("transistor saturation",
                            "A saturated transistor has a collector to emitter voltage of about 0.2 volts."),
        ("base emitter",    "A conducting silicon transistor has a base to emitter voltage of about 0.7 volts."),
    ]

    # ---- digital and interfaces ------------------------------------------
    F += [
        ("i2c",             "I2C uses two wires, SDA and SCL, and both need pull-up resistors."),
        ("i2c speed",       "Standard I2C runs at 100 kHz and fast mode runs at 400 kHz."),
        ("spi",             "SPI uses four wires: MOSI, MISO, SCK and a chip select per device."),
        ("uart",            "UART uses two wires and no clock, so both ends must agree on the baud rate."),
        ("common baud",     "Common baud rates are 9600, 57600 and 115200."),
        ("byte",            "One byte is 8 bits."),
        ("adc resolution",  "An n-bit converter has 2 to the power n levels."),
        ("twelve bit adc",  "A 12-bit converter has 4096 levels."),
        ("ten bit adc",     "A 10-bit converter has 1024 levels."),
        ("adc step",        "The ADC step size equals the reference voltage divided by the number of levels."),
        ("nyquist",         "A signal must be sampled at more than twice its highest frequency."),
        ("logic levels",    "For 3.3 volt logic, an input above about 2 volts reads high and below about 0.8 volts reads low."),
        ("pull up resistor","A typical pull-up resistor is between 4700 and 10000 ohms."),
        ("open drain",      "An open drain output can only pull low, so it needs a pull-up to go high."),
        ("debounce",        "A mechanical switch needs about 10 to 50 milliseconds of debounce time."),
    ]

    # ---- thermodynamics ---------------------------------------------------
    F += [
        ("ideal gas law",   "Pressure times volume equals the number of moles times the gas constant times temperature."),
        ("specific heat formula",
                            "Heat equals mass times specific heat capacity times the temperature change."),
        ("water specific heat",
                            "The specific heat capacity of water is 4186 J/(kg K)."),
        ("thermal conduction",
                            "Heat flow equals thermal conductivity times area times temperature difference divided by thickness."),
        ("thermal expansion",
                            "The change in length equals the expansion coefficient times the original length times the temperature change."),
        ("latent heat fusion",
                            "Melting one kilogram of ice takes about 334000 joules."),
        ("latent heat vaporisation",
                            "Boiling one kilogram of water takes about 2260000 joules."),
        ("water boiling",   "Water boils at 100 degrees Celsius at standard atmospheric pressure."),
        ("water freezing",  "Water freezes at 0 degrees Celsius at standard atmospheric pressure."),
        ("efficiency",      "Efficiency equals useful output energy divided by total input energy."),
    ]

    # ---- mechanics --------------------------------------------------------
    F += [
        ("torque",          "Torque equals force times the perpendicular distance from the pivot."),
        ("spring force",    "Spring force equals the spring constant times the extension."),
        ("spring energy",   "Energy stored in a spring equals half the spring constant times the extension squared."),
        ("pendulum period", "A pendulum period equals 2 pi times the square root of length divided by gravity."),
        ("friction",        "Friction force equals the coefficient of friction times the normal force."),
        ("lever",           "For a balanced lever, force times distance on one side equals force times distance on the other."),
        ("gear ratio",      "The gear ratio equals the number of teeth on the driven gear divided by the driver."),
        ("centripetal force",
                            "Centripetal force equals mass times velocity squared divided by the radius."),
        ("impulse",         "Impulse equals force times time and equals the change in momentum."),
        ("mechanical advantage",
                            "Mechanical advantage equals the output force divided by the input force."),
        ("stress",          "Stress equals force divided by cross-sectional area."),
        ("strain",          "Strain equals the change in length divided by the original length."),
        ("youngs modulus",  "Young's modulus equals stress divided by strain."),
    ]

    # ---- waves and optics -------------------------------------------------
    F += [
        ("snells law",      "The refractive index times the sine of the angle is equal on both sides of a boundary."),
        ("lens equation",   "One over the focal length equals one over the object distance plus one over the image distance."),
        ("magnification",   "Magnification equals the image distance divided by the object distance."),
        ("refractive index water", "The refractive index of water is about 1.33."),
        ("refractive index glass", "The refractive index of glass is about 1.5."),
        ("sound intensity", "Sound intensity falls with the square of the distance from the source."),
        ("decibel power",   "Gain in decibels for power is 10 times the log of the power ratio."),
    ]

    # ---- material properties ----------------------------------------------
    for name, dens, melt in [
        ("water", 1000, 0), ("aluminium", 2700, 660), ("copper", 8960, 1085),
        ("steel", 7850, 1425), ("iron", 7874, 1538), ("lead", 11340, 327),
        ("gold", 19300, 1064), ("silver", 10490, 962), ("tin", 7310, 232),
        ("zinc", 7140, 420), ("nickel", 8908, 1455), ("titanium", 4506, 1668),
    ]:
        F.append((f"{name} density",
                  f"The density of {name} is {dens} kilograms per cubic metre."))
        F.append((f"{name} melting point",
                  f"The melting point of {name} is {melt} degrees Celsius."))

    F += [
        ("air density",     "The density of air at sea level is about 1.225 kilograms per cubic metre."),
        ("copper conductivity",
                            "The thermal conductivity of copper is about 401 W/(m K)."),
        ("aluminium conductivity",
                            "The thermal conductivity of aluminium is about 237 W/(m K)."),
        ("steel modulus",   "The Young's modulus of steel is about 200 gigapascals."),
        ("aluminium modulus","The Young's modulus of aluminium is about 69 gigapascals."),
        ("solder melting",  "The melting point of solder is 188 degrees Celsius."),
        ("silver resistivity",  "The resistivity of silver is 1.59e-8 ohm m at 20 C."),
        ("gold resistivity",    "The resistivity of gold is 2.44e-8 ohm m at 20 C."),
        ("iron resistivity",    "The resistivity of iron is 9.7e-8 ohm m at 20 C."),
    ]

    # ---- capacitor code markings -----------------------------------------
    for code, val in [("101","100 pF"), ("102","1 nF"), ("103","10 nF"),
                      ("104","100 nF"), ("105","1 uF"), ("106","10 uF"),
                      ("221","220 pF"), ("471","470 pF"), ("222","2.2 nF"),
                      ("473","47 nF"), ("224","220 nF")]:
        F.append((f"capacitor code {code}",
                  f"A capacitor marked {code} is {val}."))
        F.append((f"capacitor {code}", f"A capacitor marked {code} is {val}."))

    # ---- batteries --------------------------------------------------------
    F += [
        ("alkaline cell",   "An alkaline AA cell is 1.5 volts."),
        ("nimh cell",       "A NiMH cell is 1.2 volts nominal."),
        ("lithium ion cell","A lithium ion cell is 3.7 volts nominal and 4.2 volts fully charged."),
        ("lifepo4 cell",    "A LiFePO4 cell is 3.2 volts nominal and 3.65 volts fully charged."),
        ("lead acid cell",  "A lead acid cell is 2.1 volts, so a 12 volt battery has six cells."),
        ("coin cell",       "A CR2032 coin cell is 3 volts and holds about 220 milliamp hours."),
        ("battery capacity","Battery energy in watt hours equals capacity in amp hours times the voltage."),
        ("battery runtime", "Runtime in hours equals battery capacity in amp hours divided by the current draw in amps."),
    ]

    # ---- embedded reference ----------------------------------------------
    F += [
        ("esp32 s3",        "The ESP32-S3 is a dual core Xtensa LX7 running up to 240 MHz."),
        ("esp32 adc",       "The ESP32 ADC is 12 bits, giving 4096 levels."),
        ("microcontroller flash",
                            "Program code is stored in flash memory and keeps its contents without power."),
        ("sram",            "SRAM loses its contents when power is removed."),
        ("spi modes",       "The four SPI modes are set by the clock polarity and clock phase."),
        ("gpio current",    "A typical microcontroller pin should not source or sink more than about 20 milliamps."),
        ("decoupling capacitor",
                            "A 100 nF capacitor placed close to each supply pin smooths switching noise."),
        ("crystal load",    "A crystal needs its specified load capacitance to oscillate at the right frequency."),
    ]

    # ---- common parts -----------------------------------------------------
    F += [
        ("555 timer",       "The 555 timer can run as a monostable one-shot or an astable oscillator."),
        ("555 astable",     "For a 555 astable, frequency equals 1.44 divided by R1 plus twice R2, times C."),
        ("7805",            "The 7805 regulator outputs 5 volts and needs at least about 7 volts in."),
        ("ldo dropout",     "A low dropout regulator needs its input above the output by at least the dropout voltage."),
        ("1n4148",          "The 1N4148 is a small signal silicon diode rated 100 volts and 200 milliamps."),
        ("2n2222",          "The 2N2222 is an NPN transistor rated about 40 volts and 800 milliamps."),
        ("lm358",           "The LM358 is a dual op amp that runs from a single supply."),
        ("regulator power", "The power wasted in a linear regulator equals the voltage dropped times the current."),
    ]

    # ---- elements 21-36 ---------------------------------------------------
    for z, sym, name, mass in [
        (21,"Sc","scandium",44.956), (22,"Ti","titanium",47.867),
        (23,"V","vanadium",50.942),  (24,"Cr","chromium",51.996),
        (25,"Mn","manganese",54.938),(26,"Fe","iron",55.845),
        (27,"Co","cobalt",58.933),   (28,"Ni","nickel",58.693),
        (29,"Cu","copper",63.546),   (30,"Zn","zinc",65.38),
        (31,"Ga","gallium",69.723),  (32,"Ge","germanium",72.630),
        (33,"As","arsenic",74.922),  (34,"Se","selenium",78.971),
        (35,"Br","bromine",79.904),  (36,"Kr","krypton",83.798),
    ]:
        F.append((name, f"The atomic mass of {name} is {mass} atomic mass units."))
        F.append((name, f"The atomic number of {name} is {z}."))
        F.append((name, f"The symbol of {name} is {sym}."))
        F.append((f"{name} atomic mass",
                  f"The atomic mass of {name} is {mass} atomic mass units."))
        F.append((f"{name} atomic number",
                  f"The atomic number of {name} is {z}."))

    # ---- more AWG rows ----------------------------------------------------
    for awg, dia, amps in [(8,3.264,40),(10,2.588,30),(12,2.053,20),(16,1.291,10),
                           (18,1.024,7),(20,0.812,5),(22,0.644,3),(24,0.511,2)]:
        F.append((f"awg {awg}",
                  f"AWG {awg} copper wire is {dia} mm in diameter and carries about {amps} amps."))

    # ---- multi-word lookups people actually type -------------------------
    F += [
        ("specific heat", "The specific heat capacity of water is 4186 J/(kg K)."),
        ("specific heat", "Heat equals mass times specific heat capacity times the temperature change."),
        ("forward voltage", "A silicon diode has a forward voltage drop of about 0.7 volts."),
        ("forward voltage", "A red LED has a forward voltage of about 1.8 volts."),
        ("time constant", "The RC time constant equals resistance times capacitance."),
        ("chip select", "SPI uses four wires: MOSI, MISO, SCK and a chip select per device."),
        ("baud rate", "Common baud rates are 9600, 57600 and 115200."),
    ]
    return F

def core_facts():
    F = []
    add = lambda k, t: F.append((k, t))

    # ---- physical constants (SI 2019; c, h, e, k, N_A are exact by definition)
    consts = [
        ("speed of light",      "The speed of light in vacuum is 299792458 m/s."),
        ("planck constant",     "The Planck constant is 6.62607015e-34 J s."),
        ("elementary charge",   "The elementary charge is 1.602176634e-19 C."),
        ("boltzmann constant",  "The Boltzmann constant is 1.380649e-23 J/K."),
        ("avogadro number",     "The Avogadro constant is 6.02214076e23 per mol."),
        ("gravitational constant",
                                "The gravitational constant is 6.67430e-11 N m2/kg2."),
        ("standard gravity",    "Standard gravity is 9.80665 m/s2."),
        ("gas constant",        "The molar gas constant is 8.314462618 J/(mol K)."),
        ("electron mass",       "The electron rest mass is 9.1093837015e-31 kg."),
        ("proton mass",         "The proton rest mass is 1.67262192369e-27 kg."),
        ("neutron mass",        "The neutron rest mass is 1.67492749804e-27 kg."),
        ("vacuum permittivity", "The vacuum permittivity is 8.8541878128e-12 F/m."),
        ("vacuum permeability", "The vacuum permeability is 1.25663706212e-6 H/m."),
        ("stefan boltzmann",    "The Stefan-Boltzmann constant is 5.670374419e-8 W/(m2 K4)."),
        ("absolute zero",       "Absolute zero is 0 K, which is -273.15 degrees Celsius."),
        ("atmospheric pressure","Standard atmospheric pressure is 101325 Pa."),
        ("speed of sound",      "The speed of sound in dry air at 20 C is about 343 m/s."),
    ]
    F += consts

    # ---- EE and physics formulas
    formulas = [
        ("ohms law",        "Voltage equals current times resistance."),
        ("electrical power","Power equals voltage times current."),
        ("power resistance","Power equals current squared times resistance."),
        ("series resistance","Series resistance is the sum of the resistances."),
        ("parallel resistance",
                            "For two resistors in parallel, resistance is the product divided by the sum."),
        ("series capacitance",
                            "For two capacitors in series, capacitance is the product divided by the sum."),
        ("parallel capacitance",
                            "Parallel capacitance is the sum of the capacitances."),
        ("capacitive reactance",
                            "Capacitive reactance equals 1 divided by 2 pi f C."),
        ("inductive reactance",
                            "Inductive reactance equals 2 pi f L."),
        ("resonant frequency",
                            "Resonant frequency equals 1 divided by 2 pi times the square root of L times C."),
        ("rc time constant","The RC time constant equals resistance times capacitance."),
        ("rl time constant","The RL time constant equals inductance divided by resistance."),
        ("capacitor energy","Energy in a capacitor equals half C times voltage squared."),
        ("inductor energy", "Energy in an inductor equals half L times current squared."),
        ("charge",          "Charge equals current times time."),
        ("kirchhoff current",
                            "The currents entering a node equal the currents leaving it."),
        ("kirchhoff voltage",
                            "The voltages around a closed loop sum to zero."),
        ("transformer ratio",
                            "The voltage ratio of a transformer equals its turns ratio."),
        ("rms voltage",     "For a sine wave, RMS voltage is the peak voltage divided by the square root of 2."),
        ("decibel",         "Gain in decibels is 20 times the log of the voltage ratio."),
        ("wire resistance", "Wire resistance equals resistivity times length divided by area."),
        ("copper resistivity",
                            "The resistivity of copper is 1.68e-8 ohm m at 20 C."),
        ("aluminium resistivity",
                            "The resistivity of aluminium is 2.65e-8 ohm m at 20 C."),
        # mechanics
        ("newtons second law", "Force equals mass times acceleration."),
        ("kinetic energy",  "Kinetic energy equals half mass times velocity squared."),
        ("potential energy","Gravitational potential energy equals mass times gravity times height."),
        ("momentum",        "Momentum equals mass times velocity."),
        ("work",            "Work equals force times distance."),
        ("mechanical power","Power equals work divided by time."),
        ("pressure",        "Pressure equals force divided by area."),
        ("density",         "Density equals mass divided by volume."),
        ("velocity",        "Velocity equals distance divided by time."),
        ("acceleration",    "Acceleration equals change in velocity divided by time."),
        ("wave speed",      "Wave speed equals frequency times wavelength."),
        ("ohm",             "One ohm is one volt per ampere."),
        ("watt",            "One watt is one joule per second."),
        ("farad",           "One farad is one coulomb per volt."),
        ("henry",           "One henry is one weber per ampere."),
    ]
    F += formulas

    # ---- SI prefixes
    for name, sym, val in [("tera","T","1e12"), ("giga","G","1e9"),
                           ("mega","M","1e6"), ("kilo","k","1e3"),
                           ("milli","m","1e-3"), ("micro","u","1e-6"),
                           ("nano","n","1e-9"), ("pico","p","1e-12")]:
        add(f"{name} prefix", f"The prefix {name} ({sym}) means {val}.")

    # ---- unit conversions
    conv = [
        ("inch",        "One inch is 25.4 millimetres."),
        ("foot",        "One foot is 0.3048 metres."),
        ("mile",        "One mile is 1.609344 kilometres."),
        ("pound",       "One pound is 0.45359237 kilograms."),
        ("gallon",      "One US gallon is 3.785411784 litres."),
        ("horsepower",  "One mechanical horsepower is about 745.7 watts."),
        ("calorie",     "One calorie is 4.184 joules."),
        ("electronvolt","One electronvolt is 1.602176634e-19 joules."),
        ("bar",         "One bar is 100000 pascals."),
        ("psi",         "One psi is about 6894.76 pascals."),
        ("knot",        "One knot is 1.852 kilometres per hour."),
        ("celsius to fahrenheit",
                        "Fahrenheit equals Celsius times 9 divided by 5 plus 32."),
        ("kelvin",      "Kelvin equals degrees Celsius plus 273.15."),
    ]
    F += conv

    # ---- E24 resistor series + colour code
    e24 = [10,11,12,13,15,16,18,20,22,24,27,30,33,36,39,43,47,51,56,62,68,75,82,91]
    add("e24 series", "The E24 resistor series values are " +
        ", ".join(str(v) for v in e24) + ".")
    add("e12 series", "The E12 resistor series values are 10, 12, 15, 18, 22, "
                      "27, 33, 39, 47, 56, 68, 82.")
    colours = ["black","brown","red","orange","yellow",
               "green","blue","violet","grey","white"]
    for i, c in enumerate(colours):
        add(f"{c} band", f"A {c} resistor band means the digit {i}.")
    add("gold band", "A gold tolerance band means 5 percent.")
    add("silver band", "A silver tolerance band means 10 percent.")

    # ---- wire gauge (copper, chassis-style rough ampacity)
    for awg, dia, amps in [(10,2.588,30),(12,2.053,20),(14,1.628,15),
                           (16,1.291,10),(18,1.024,7),(20,0.812,5),
                           (22,0.644,3),(24,0.511,2)]:
        add(f"awg {awg}", f"AWG {awg} copper wire is {dia} mm in diameter and "
                          f"carries about {amps} amps.")

    # ---- elements 1-20
    elements = [
        (1,"hydrogen","H",1.008),(2,"helium","He",4.0026),(3,"lithium","Li",6.94),
        (4,"beryllium","Be",9.0122),(5,"boron","B",10.81),(6,"carbon","C",12.011),
        (7,"nitrogen","N",14.007),(8,"oxygen","O",15.999),(9,"fluorine","F",18.998),
        (10,"neon","Ne",20.180),(11,"sodium","Na",22.990),(12,"magnesium","Mg",24.305),
        (13,"aluminium","Al",26.982),(14,"silicon","Si",28.085),(15,"phosphorus","P",30.974),
        (16,"sulfur","S",32.06),(17,"chlorine","Cl",35.45),(18,"argon","Ar",39.948),
        (19,"potassium","K",39.098),(20,"calcium","Ca",40.078),
    ]
    for z, name, sym, mass in elements:
        add(name, f"The atomic mass of {name} is {mass} atomic mass units.")
        add(name, f"The atomic number of {name} is {z}.")
        add(name, f"The symbol of {name} is {sym}.")
        # also state each value in the canonical "The X of Y is Z" shape, which
        # is what the model was trained to extract from
        add(f"{name} atomic mass",
            f"The atomic mass of {name} is {mass} atomic mass units.")
        add(f"{name} atomic number", f"The atomic number of {name} is {z}.")

    # ---- solar system (approximate, widely published figures)
    planets = [
        ("mercury", 4879, 57.9, 88),   ("venus", 12104, 108.2, 225),
        ("earth", 12742, 149.6, 365),  ("mars", 6779, 227.9, 687),
        ("jupiter", 139820, 778.5, 4333), ("saturn", 116460, 1434, 10759),
        ("uranus", 50724, 2871, 30687),   ("neptune", 49244, 4495, 60190),
    ]
    for name, dia, dist, year in planets:
        add(name, f"{name.capitalize()} has a diameter of about {dia} km, orbits "
                  f"about {dist} million km from the Sun, and takes about {year} "
                  f"Earth days to orbit.")
    add("sun", "The Sun has a diameter of about 1392700 km.")
    add("moon", "The Moon has a diameter of about 3475 km and orbits about "
                "384400 km from Earth.")

    F += extended_facts()
    F += my_facts()
    # multi-word keys also get pair aliases: "aluminium melting point" becomes
    # reachable as "aluminium melting" and "aluminium point", because the
    # device only forms two-word phrases from a question.
    pair_alias = []
    for k, t in F:
        w = k.split()
        if len(w) >= 3:
            for i in range(len(w)):
                for j in range(i + 1, len(w)):
                    pair_alias.append((f"{w[i]} {w[j]}", t))
    F += pair_alias
    F += formula_aliases(F)
    return F

# ===========================================================================
# OPTIONAL CONVERTERS (run these on a machine with internet)
# ===========================================================================

def from_wordnet(path):
    """WordNet data.noun -> 'All Xs are Ys.' hypernym facts.
    Get it with:  pip install nltk; python -c "import nltk;nltk.download('wordnet')"
    then point at .../wordnet/data.noun"""
    out, offsets = [], {}
    lines = [l for l in open(path, encoding="latin-1") if not l.startswith("  ")]
    for l in lines:                                  # first pass: offset -> word
        p = l.split()
        if len(p) > 4:
            offsets[p[0]] = p[4].replace("_", " ")
    for l in lines:
        p = l.split()
        if len(p) < 5:
            continue
        word = p[4].replace("_", " ")
        for i, tok in enumerate(p):
            if tok == "@" and i + 1 < len(p):        # @ = hypernym pointer
                parent = offsets.get(p[i + 1])
                if parent and parent != word:
                    out.append((word.lower(), f"All {word}s are {parent}s."))
                break
    return out

def from_factbook(path):
    """factbook.json (github.com/factbook/factbook.json) -> country facts.
    US government work, public domain."""
    out = []
    data = json.load(open(path, encoding="utf-8"))
    countries = data if isinstance(data, list) else [data]
    for c in countries:
        try:
            name = c["Government"]["Country name"]["conventional short form"]["text"]
            pop = c["People and Society"]["Population"]["total"]["text"]
            area = c["Geography"]["Area"]["total"]["text"]
            out.append((name.lower(), f"{name} has a population of {pop}."))
            out.append((name.lower(), f"{name} has an area of {area}."))
        except (KeyError, TypeError):
            continue
    return out

def from_csv(path):
    out = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        k, _, t = line.partition(",")
        if t:
            out.append((k.strip().lower(), t.strip()))
    return out

# ===========================================================================
# PACKING
# ===========================================================================


STOP = {"a","an","the","of","in","to","and","is","are","per","for","by","on"}

def expand_aliases(facts):
    """Also index each significant word of a multi-word key, so a query for
    'copper' finds 'copper resistivity'. Costs index space, buys recall."""
    out = list(facts)
    for k, t in facts:
        parts = [w for w in k.split() if w not in STOP and len(w) > 2]
        if len(parts) > 1:
            for w in parts:
                out.append((w, t))
    return out

def build(facts, out_path):
    facts = expand_aliases(facts)
    grouped = defaultdict(list)
    for k, t in facts:
        k = k.strip().lower()[:KEY_BYTES]
        if k and t and t not in grouped[k]:
            grouped[k].append(t)

    keys = sorted(grouped)                       # bytewise sort == device sort
    records, index, off = bytearray(), bytearray(), 0
    header = 16
    for k in keys:
        kb = k.encode("utf-8")[:KEY_BYTES]
        rec = bytearray()
        rec += struct.pack("<B", len(kb)) + kb
        texts = grouped[k][:6]                   # cap facts per key
        rec += struct.pack("<B", len(texts))
        for t in texts:
            tb = t.encode("utf-8")[:400]
            rec += struct.pack("<H", len(tb)) + tb
        index += kb.ljust(KEY_BYTES, b"\0") + struct.pack("<IH", header + off,
                                                          len(rec))
        records += rec
        off += len(rec)

    idx_off = header + len(records)
    with open(out_path, "wb") as f:
        f.write(b"R457KB01" + struct.pack("<II", len(keys), idx_off))
        f.write(records)
        f.write(index)
    return len(keys), sum(len(v) for v in grouped.values()), os.path.getsize(out_path)

def query(path, key):
    """Binary search — mirrors exactly what the ESP32 firmware will do."""
    f = open(path, "rb")
    magic, n, idx_off = struct.unpack("<8sII", f.read(16))
    assert magic == b"R457KB01", "not a kb.bin"
    target = key.strip().lower()[:KEY_BYTES].encode("utf-8")
    lo, hi, seeks = 0, n - 1, 0
    while lo <= hi:
        mid = (lo + hi) // 2
        f.seek(idx_off + mid * IDX_ENTRY); seeks += 1
        entry = f.read(IDX_ENTRY)
        k = entry[:KEY_BYTES].rstrip(b"\0")
        if k == target:
            rec_off, rec_len = struct.unpack("<IH", entry[KEY_BYTES:])
            f.seek(rec_off); rec = f.read(rec_len); seeks += 1
            p = 1 + rec[0]
            nf = rec[p]; p += 1
            out = []
            for _ in range(nf):
                ln = struct.unpack("<H", rec[p:p+2])[0]; p += 2
                out.append(rec[p:p+ln].decode("utf-8")); p += ln
            return out, seeks
        if k < target: lo = mid + 1
        else:          hi = mid - 1
    return [], seeks

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="kb.bin")
    ap.add_argument("--wordnet"); ap.add_argument("--factbook"); ap.add_argument("--csv")
    ap.add_argument("--query")
    a = ap.parse_args()

    if a.query:
        facts, seeks = query(a.out, a.query)
        print(f"query {a.query!r}: {len(facts)} fact(s) in {seeks} seeks")
        for t in facts: print("  " + t)
        return

    facts = core_facts()
    print(f"authored core: {len(facts)} facts")
    for flag, fn, label in [(a.wordnet, from_wordnet, "wordnet"),
                            (a.factbook, from_factbook, "factbook"),
                            (a.csv, from_csv, "csv")]:
        if flag:
            got = fn(flag)
            print(f"{label}: +{len(got)} facts")
            facts += got

    keys, total, size = build(facts, a.out)
    import math
    print(f"\nwrote {a.out}: {keys:,} keys, {total:,} facts, {size/1e6:.2f} MB")
    print(f"lookup cost on device: about {math.ceil(math.log2(max(2,keys)))+1} "
          f"SD reads per query")

if __name__ == "__main__":
    main()
