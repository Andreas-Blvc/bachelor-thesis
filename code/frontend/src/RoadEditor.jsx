import { useState, useEffect, useRef, useCallback } from "react";
import axios from "axios";
import * as fabric from 'fabric';

const RoadEditor = () => {
  const [selectedRoad, setSelectedRoad] = useState("StraightRoad");
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!canvasRef.current) {
      console.log('Initializing canvas');
      canvasRef.current = new fabric.Canvas('roadCanvas', {
        height: 400,
        width: 600,
        backgroundColor: 'white',
        selection: false
      });
      canvasRef.current.re;
    }
    return () => {
      if (canvasRef.current) {
        console.log('Disposing canvas');
        canvasRef.current.dispose();
        canvasRef.current = null;
      }
    };
  }, []);

  const addRoadSegment = useCallback((x, y, roadType) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    console.log("Adding Road Segment", roadType);

    if (roadType === "StraightRoad") {
      const length = parseFloat(prompt("Enter the length of the Straight Road:")) || 100;
      const directionAngle = parseFloat(prompt("Enter the direction angle (in degrees):")) || 0;
      const radians = (directionAngle * Math.PI) / 180;

      const x2 = x + length * Math.cos(radians);
      const y2 = y + length * Math.sin(radians);

      const line = new fabric.Line([x, y, x2, y2], {
        stroke: "black",
        strokeWidth: 3,
        selectable: false,
        evented: false
      });
      canvas.add(line);
    } else if (roadType === "CircularCurve") {
      const radius = parseFloat(prompt("Enter the radius of the Circular Curve Road:")) || 50;
      const startAngle = parseFloat(prompt("Enter the start angle (in degrees):")) || 0;
      const angleSweep = parseFloat(prompt("Enter the angle sweep (in degrees):")) || 90;

      const radiansStart = (startAngle * Math.PI) / 180;
      const radiansSweep = (angleSweep * Math.PI) / 180;

      const arcPath = `M ${x + radius * Math.cos(radiansStart)} ${y + radius * Math.sin(radiansStart)}
                       A ${radius} ${radius} 0 ${angleSweep > 180 ? 1 : 0} 1 
                       ${x + radius * Math.cos(radiansStart + radiansSweep)} ${y + radius * Math.sin(radiansStart + radiansSweep)}`;

      const arc = new fabric.Path(arcPath, {
        stroke: "blue",
        fill: "transparent",
        strokeWidth: 3,
        selectable: false,
        evented: false
      });
      canvas.add(arc);
    }
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const handleMouseDown = (event) => {
      if (!event.pointer) return;
      const { x, y } = event.pointer;
      addRoadSegment(x, y, selectedRoad);
    };

    canvas.off("mouse:down");
    canvas.on("mouse:down", handleMouseDown);

    return () => {
      canvas.off("mouse:down", handleMouseDown);
    };
  }, [selectedRoad, addRoadSegment]);

  const saveRoad = async () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const roadData = JSON.stringify(canvas.toJSON());

    try {
      await axios.post("http://localhost:5000/save_road", {
        filename: "user_road",
        roads: roadData,
      });
      alert("Road saved successfully!");
    } catch (error) {
      console.error("Error saving road:", error);
    }
  };

  return (
    <div>
      <h2>Road Editor</h2>
      <div>
        <button onClick={() => setSelectedRoad("StraightRoad")}>Straight Road</button>
        <button onClick={() => setSelectedRoad("CircularCurve")}>Circular Curve</button>
        <button onClick={saveRoad}>Save Road</button>
      </div>
      <canvas id="roadCanvas" style={{ border: "1px solid black"}}></canvas>
    </div>
  );
};

export default RoadEditor;
