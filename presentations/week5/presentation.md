# Woche 5

---

## Was habe ich letzte Woche gemacht?

- Orientierung und Lenkwinkel modelliert.
- Constraints linearisiert:
    - Vergleich zwischen Taylor- und McCormick-Ansatz.
    - Ergebnis: McCormick funktioniert besser, insbesondere bei eingeschränkter Fahrzeuggeschwindigkeit.

----

## McCormick envelopes

![McCormick envelopes](./resources/Figure_4.png)

---

## Welche Fragen oder Probleme sind aufgetreten?

- Das Haupt-Paper basiert stark darauf, dass State-Variablen unabhängig voneinander gesteuert werden können:
	- Umsetzung von Orientierung und Lenkwinkel nach dem Ansatz aus dem Paper schwierig.
- Neuer Linearisierungsansatz:
	- Keine Garantie auf Feasibility.
    - Aber: berechnete Lösungen liegen nah an einer feasible Lösung.
- Idee: Nachträgliche Korrektur durch ein nicht-konvexes Optimierungsproblem.

---

## Was möchte ich nächste Woche tun?

- Weiteres Testen des neuen Ansatzes.
- Ansätze entwickeln, um Feasibility zu garantieren.
- Gantt-Chart überarbeiten.