# Woche 3

---

## Was habe ich letzte Woche gemacht?

- Core Paper
	- Neues Modell im Frenet-Frame
	- Berechnung der Curvature
	- Implementierung der Constraints
- Mosek-Modellierung
- Fokus auf das Point-Mass-Modell und Straßen-Constraints

----

![Path Planning Visualization](./resources/Figure_1.png)


----

![Curvature](./resources/Figure_2.png)

---

## Welche Fragen oder Probleme sind aufgetreten?

- Annahmen im Core Paper:
    - Die Curvature ist linear oder konstant
    - Die Orientierung des Autos ist stets parallel zur Straße
- '∀' in der Unterapproximation der Constraints durch konvexe Teilmengen

----

![Under Approximation](./resources/Figure_3.png)

---

## Was möchte ich nächste Woche tun?

- Implementierung der '∀'-Eliminierung, um cvxpy-kompatibel zu sein
- Einschränkungen der Annahmen genauer analysieren
- Weitere relevante Papers finden