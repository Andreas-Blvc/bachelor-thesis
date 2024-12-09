# Woche 8

---

## Was habe ich letzte Woche gemacht?

- Viel im Doing:
  - RoadEditor gebaut, um einfach und schnell viele Szenarien zu testen
  - Straße besteht aus einzelnen Segmenten
    - erlaubt piecewise linear curvature, piecewise concave (convex) laterel-offset upper (lower) bound
  - Implementierung Benchmarks, konfiguration durch Standard Menge an Parameterwerten (leicht erweiterbar), darunter:
  	- start_velocity, start_lateral_offset, velocity_range, road, time_horizon, time_discretization, models, objectiv
  - Path-Planning mit Segment-Übergängen, aktuell: 
    - Plane bis zum nächsten Segment und setze neuen initialen Zustand, wechsle Segment, wiederhole 


----

- Theorie:
  - Paper Gutjahr: 
    - nutzt ebenfalls Winkelannäherung und 1-nC(s)≈1
    - Kein Lenkwinkel, stattdessen: d_psi = v * yaw_rate, Änderung der yaw_rate als input
  - Erkenntnis Benchmarking: 
    - Mein Ansatz: 
      - Winkel-Annährung -> Orientierungsänderung stärker als geplant 
      - McCormick-Annäherung -> oft |xy| <= w -> state-änderungen größer als sie sein sollten
    - Main-Paper: 
      - keine Approximationsfehler, allerdings: Seitwärts Bewegung möglich, kein Lenkwinkel, Zustandsraum eingeschränkt

---

## Welche Fragen oder Probleme sind aufgetreten?

- Wie analysiert man datenbasiert und macht konkrete Aussagen?
- Wie stelle ich sicher, dass der Algorithmus robust gegenüber unterschiedlichen Szenarien bleibt?


---

## Was möchte ich nächste Woche tun?

- Path Planning als Online-Algorithmus umsetzen
  - ermöglicht bessere Approximation
- Related Work
