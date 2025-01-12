# Woche 10

---

## Was habe ich letzte Woche gemacht?

- Auswertung der Modelle, siehe GIFs
- Umsetzung/Implementierung der lokalen Konvexifizierung
- Eq. 24: Krümmung approximieren, siehe Ausformulierung

---

## Welche Fragen oder Probleme sind aufgetreten?

- Numerische Instabilität bei niedrigen Geschwindigkeiten (< 2 m/s)
- Der Double-Integrator schneidet aktuell noch nicht so gut ab.
- Die Simulation dauert lange, da cvxpy Probleme teilweise mit einer Dauer von > 1 s kompiliert. Durch das ständige Re-Planning summiert sich die Dauer.

---

## Was möchte ich nächste Woche tun?

- Aktuelles Vorgehen aufschreiben (KST) mit möglichen Verbesserungen
- Double-Integrator verbessern:
	- Inneres Polytop vergrößern (für alle Operatoren)
	- Übersetzung der Planning Controls (Point Mass) auf Vehicle Controls (Dynamic Single Track)


