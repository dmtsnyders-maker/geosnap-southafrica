# Methodology

## Data
| Layer | Source | Notes |
|---|---|---|
| Ward population, age (5-year bands), sex, population group | Stats SA ward product (2022, modelled small-area estimates) | One vintage only; fractional counts |
| Ward boundaries | Municipal Demarcation Board "MDB Wards 2020" (4,468 wards) | `WardID` = Stats SA `Ward_Code` |
| Yearly population 2015-2030 | WorldPop Global2 R2025A, 100 m, constrained | Modelled; used only to rescale 2022 |
| Suburbs | OpenStreetMap (ODbL) points, City of Cape Town polygons, or user supplied | Coverage of OSM varies |

## Yearly series
`pop_year(ward) = pop_2022(ward) x WP_year(ward) / WP_2022(ward)` where `WP` is the sum of the WorldPop raster
inside the ward. All count columns (population, age bands, sex) share the factor; race and age *shares* are
held at 2022. Wards with < 5 people in the anchor raster use their municipality's ratio. These are model
outputs, not census counts; years after the latest observations are projections. `population_basis` marks
which rows are which.

## Suburbs
Wards and suburbs do not nest. Polygon suburb layers use an equal-area overlay (share of ward, share of suburb;
a suburb is listed if >= 5% of the ward or >= 50% of the suburb lies in it). Point layers (OSM) use point-in-ward,
nearest the ward centre first. `interpolate_to_suburbs` uses area-weighted interpolation (tobler) and assumes even
density inside each ward.
