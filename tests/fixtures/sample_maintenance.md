# Turbine Engine Maintenance Guide

## Overview

This guide covers standard maintenance procedures for the PW1100G-JM geared turbofan engine. All procedures must be performed by certified technicians in accordance with the Engine Maintenance Manual (EMM).

## Fan Module

### Fan Blade Inspection

Fan blades shall be inspected at every shop visit using the following methods:

1. **Visual inspection** — Check for leading edge erosion, FOD damage, and tip curl
2. **Fluorescent Penetrant Inspection (FPI)** — Required for all blades exceeding 10,000 cycles
3. **Dimensional check** — Measure chord width at 25%, 50%, and 75% span stations

#### Erosion Limits

Leading edge erosion is acceptable if:
- Depth does not exceed 0.030 inches at any point
- Width does not exceed 0.250 inches
- No more than 3 adjacent blades are affected

Blades exceeding these limits require blend repair per SB-72-0412 or replacement.

### Fan Disk Life Limits

The fan disk is a life-limited part with a certified life of 20,000 cycles. No extensions or waivers are permitted. Upon reaching the life limit, the disk must be retired from service regardless of condition.

## High Pressure Turbine (HPT)

### Thermal Barrier Coating Assessment

HPT blades operate at temperatures exceeding 2,000°F and rely on thermal barrier coatings (TBC) for thermal protection. TBC condition is assessed via borescope inspection at intervals defined in the on-condition maintenance program.

#### TBC Spallation Criteria

- Less than 10% area loss: serviceable, monitor at next inspection
- 10-30% area loss: serviceable with reduced inspection interval (50% of normal)
- Greater than 30% area loss: unserviceable, requires strip and recoat per EMM 72-51-01

### HPT Blade Creep Assessment

High-pressure turbine blades are subject to creep deformation under sustained centrifugal and thermal loads. Creep is assessed by measuring blade tip clearance and comparing to new-condition baseline.

A creep rate exceeding 0.001 inches per 1,000 cycles indicates accelerated degradation and requires removal for metallographic evaluation.

## Bearing System

### Oil Analysis Monitoring

The Spectrometric Oil Analysis Program (SOAP) provides early warning of bearing distress:

| Element | Normal (ppm) | Watch (ppm) | Action (ppm) |
|---------|-------------|-------------|--------------|
| Iron (Fe) | < 5 | 5-10 | > 10 |
| Chromium (Cr) | < 2 | 2-5 | > 5 |
| Silver (Ag) | < 1 | 1-3 | > 3 |

Magnetic chip detector inspection is required whenever SOAP results exceed watch limits.

## Corrective Actions

### Blend Repair Procedure

Blend repairs are permitted for minor FOD and erosion damage subject to the following constraints:
- Maximum blend depth: 0.040 inches
- Maximum blend length: 0.500 inches
- Minimum remaining wall thickness: 60% of nominal
- Blend radius must be at least 10:1 (length:depth)

All blends must be polished to 32 microinch Ra or better surface finish.

### Component Replacement Decision Logic

The following decision tree governs repair-vs-replace:
1. Is the defect within blend limits? → Blend repair
2. Is the component within weld repair criteria? → Weld repair (shop visit required)
3. Is the component life-limited and approaching limit? → Replace (new or serviceable)
4. None of the above → Engineering disposition required
