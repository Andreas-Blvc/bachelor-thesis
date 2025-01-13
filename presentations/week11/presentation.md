# Woche 11

---

## Was habe ich letzte Woche gemacht?

- Übersetzung der Planning Controls (Point Mass) auf Vehicle Controls (Dynamic Single Track)
	- Verschiedene Ansätze durchprobiert
	- Am Besten: 
		- Berechnung der Krümmungsrate mittels Main Paper
		- Berechnung Lenkwinkel mittels KST dynamics
		- Berechnung Lenkwinkelrate mittels Differenz zum aktuellen Lenkwinkel 
- Scipy Diskretisierung der Dynamics.
	- Bisher: Forward Euler selbst implementiert

----

![Control Inputs](./resources/Figure_5.png)

---

## Welche Fragen oder Probleme sind aufgetreten?

- 'Lag' (Verzögerung) bei den Control Inputs wahrgenommen
	- Geplante Änderung treten durch diskretisierung später auf als erwartet
	- Tritt auf, wenn man zum Übersetzen der Control Inputs, die Formeln von Forward Euler verwendet
- Double Integrator Constraints zu stark

---

## Was möchte ich nächste Woche tun?

- Double Integrator Constraints verbessern
	- Mehr Szenarien schaffen