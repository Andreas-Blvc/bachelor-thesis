# Woche 9

---

## Was habe ich letzte Woche gemacht?

- Simulationsmodell implementiert (Single-Track Model (ST))
- Berechnung der Steuerwinkelgeschwindigkeit und der Längsbeschleunigung des Fahrzeugmodells ST mithilfe der aktuellen Planungsmodelle
- Erste erfolgreiche Fahrt auf doppeltem Spurwechsel
  - Aktuelle Fahrvorschrift: Plane die nächste Sekunde mit 60 Diskretisierungspunkten (gleichverteilt) in <10 ms, fahre 50 ms mit geplanten Steuereingaben, wiederhole bis Ende erreicht
- Benchmarking-Framework somit fertiggestellt

---

## Welche Fragen oder Probleme sind aufgetreten?

- Aktuelle Berechnung der Steuerwinkelgeschwindigkeit mit Double-Integrator-Modell nicht genau genug
- cvxpy: Ständige Kompilierung des Problems bremst mich beim Testen aus -> Simulationen, in denen das Auto 10 s fährt, brauchen ~10 min zum Erstellen, obwohl die reine Rechenzeit des Solvers deutlich geringer ist

---

## Was möchte ich nächste Woche tun?

- Parametrisierung des Optimierungsproblems (cvxpy)
- Planungsmodelle verbessern
  - Berechnung der Steuerwinkelgeschwindigkeit
  - Taylor- und McCormick-Approximationen um die zuletzt berechneten Werte legen
- bessere Objectives definieren
  - Aktuell funktioniert maximize_distance_traveled bei dem kinematischen Single-Track-Planungsmodell gut

