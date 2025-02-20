from wolframclient.evaluation import WolframLanguageSession
from wolframclient.exception import WolframKernelException
from wolframclient.language import wlexpr, wl
from sympy import symbols, And, Or, Not, Eq, Ne, Ge, Le, Lt, Gt, Add, Mul, Pow
import time
import subprocess

def kill_wolfram_kernel():
    try:
        # Step 1: List all processes and filter for WolframKernel
        result = subprocess.run(['pgrep', 'WolframKernel'], capture_output=True, text=True)

        # Step 2: Get process IDs (PIDs)
        pids = result.stdout.strip().split('\n')

        if not pids or pids == ['']:
            print("No WolframKernel processes found.")
            return

        print(f"Found WolframKernel processes: {pids}")

        # Step 3: Kill each process
        for pid in pids:
            subprocess.run(['kill', '-9', pid], check=True)
            print(f"Killed process with PID: {pid}")

    except Exception as e:
        print(f"Error killing WolframKernel processes: {e}")



def eliminate_quantifier(
    Curvature = 1/400,
    sMin = 0,
    sMax = 10,
    nMin = -0,
    nMax = 2,
    vxMin = 0,
    vxMax = 10,
    vyMin = -2,
    vyMax = 2,
    axMin = -3,
    axMax = 6,
    ayMin = -4,
    ayMax = 4,
    dpsiMin = -5,
    dpsiMax = 5,
    apsiMin = -2,
    apsiMax = 2,
):
    # print(
    #     f"Curvature={Curvature};\n"
    #     f"sMin={sMin};\n"
    #     f"sMax={sMax};\n"
    #     f"nMin={nMin};\n"
    #     f"nMax={nMax};\n"
    #     f"vxMin={vxMin};\n"
    #     f"vxMax={vxMax};\n"
    #     f"vyMin={vyMin};\n"
    #     f"vyMax={vyMax};\n"
    #     f"axMin={axMin};\n"
    #     f"axMax={axMax};\n"
    #     f"ayMin={ayMin};\n"
    #     f"ayMax={ayMax};\n"
    #     f"dpsiMin={dpsiMin};\n"
    #     f"dpsiMax={dpsiMax};\n"
    #     f"apsiMin={apsiMin};\n"
    #     f"apsiMax={apsiMax};"
    # )

    u1, u2, x1, x2, x3, x4 = symbols('u1 u2 x1 x2 x3 x4')

    def _convert_wl_to_sympy(expr):
        """Convert Wolfram Language expression to SymPy."""

        # Handle Wolfram expressions with a head and arguments
        if hasattr(expr, 'head'):
            head = expr.head.name
            args = expr.args

            # Logical and comparison operations
            if head == "And":
                return And(*(_convert_wl_to_sympy(arg) for arg in args))
            elif head == "Or":
                return Or(*(_convert_wl_to_sympy(arg) for arg in args))
            elif head == "Not":
                return Not(_convert_wl_to_sympy(args[0]))
            elif head == "Equal":
                return Eq(_convert_wl_to_sympy(args[0]), _convert_wl_to_sympy(args[1]))
            elif head == "Unequal":
                return Ne(_convert_wl_to_sympy(args[0]), _convert_wl_to_sympy(args[1]))
            elif head == "LessEqual":
                return Le(_convert_wl_to_sympy(args[0]), _convert_wl_to_sympy(args[1]))
            elif head == "GreaterEqual":
                return Ge(_convert_wl_to_sympy(args[0]), _convert_wl_to_sympy(args[1]))
            elif head == "Less":
                return Lt(_convert_wl_to_sympy(args[0]), _convert_wl_to_sympy(args[1]))
            elif head == "Greater":
                return Gt(_convert_wl_to_sympy(args[0]), _convert_wl_to_sympy(args[1]))

            # Arithmetic operations
            elif head == "Plus":
                return Add(*(_convert_wl_to_sympy(arg) for arg in args))
            elif head == "Times":
                return Mul(*(_convert_wl_to_sympy(arg) for arg in args))
            elif head == "Power":
                return Pow(_convert_wl_to_sympy(args[0]), _convert_wl_to_sympy(args[1]))

        # Handle constants and numbers
        elif isinstance(expr, (int, float)):
            return expr

        # Handle global variables (e.g., Global`x3)
        elif 'Global' in str(expr):
            var_name = str(expr).split("Global`")[-1]
            return symbols(var_name)

        # Handle basic numbers and constants
        elif isinstance(expr, (int, float)):
            return expr

        # If not recognized, raise an error
        raise ValueError(f"Unsupported expression type: {expr}")

    session = WolframLanguageSession('/Applications/Wolfram.app/Contents/MacOS/WolframKernel')
    for _ in range(10):
        try:
            session.evaluate(wlexpr('ClearAll["Global`*"]'))
            expressions = session.evaluate(wlexpr(f'''
                constraintsZ = And[
                   {vxMin} <= x3*(1 - x2*{Curvature}),
                   x3*(1 - x2*{Curvature}) <= {vxMax},
                   {dpsiMin} <= {Curvature}*x3,
                   {Curvature}*x3 <= {dpsiMax},
                   {apsiMin} <= {Curvature}*u1,
                   {Curvature}*u1 <= {apsiMax},
                   {axMin} <= (1 - x2*{Curvature})*u1 - 2*x4*{Curvature}*x3,
                   (1 - x2*{Curvature})*u1 - 2*x4*{Curvature}*x3 <= {axMax},
                   {ayMin} <= u2 + {Curvature}*x3^2*(1 - x2*{Curvature}),
                   u2 + {Curvature}*x3^2*(1 - x2*{Curvature}) <= {ayMax}
                ];
        
                step1 = Resolve[ForAll[x1, {sMin} <= x1 <= {sMax} \[Implies] constraintsZ], 
                   Reals];
                step2 = Resolve[ForAll[x2, {nMin} <= x2 <= {nMax} \[Implies] step1], 
                   Reals];
                step3 = Resolve[ForAll[x4, {vyMin} <= x4 <= {vyMax} \[Implies] step2], 
                   Reals];
        
                ''' + '''
                boundedRegion = ImplicitRegion[step3, {u1, u2, x3}];
                
                samplePoints = RandomReal[{-10, 10}, {10000, 3}];
                validPoints = Select[samplePoints, RegionMember[boundedRegion, #] &];
                
                res = If[validPoints === {}, {},
                
                {x3Min, x3Max} = {Min[validPoints[[All, 1]]], Max[validPoints[[All, 1]]]};
                {u1Min, u1Max} = {Min[validPoints[[All, 2]]], Max[validPoints[[All, 2]]]};
                {u2Min, u2Max} = {Min[validPoints[[All, 3]]], Max[validPoints[[All, 3]]]};
                
                feasiblePoint = {x3 -> (x3Min + x3Max)/2, u1 -> (u1Min + u1Max)/2, 
                   u2 -> (u2Min + u2Max)/2};
                
                defaultDirections = Normalize /@ {
                    {1, 0, 0},
                    {0, 1, 0},
                    {0, 0, 1},
                    {-1, 0, 0},
                    {0, -1, 0},
                    {0, 0, -1},
                    {(x3Max - x3Min)/2, (u1Max - u1Min)/2, (u2Max - u2Min)/2},
                    {(x3Max - x3Min)/2, (u1Max - u1Min)/2, -(u2Max - u2Min)/2},
                    {(x3Max - x3Min)/2, -(u1Max - u1Min)/2, (u2Max - u2Min)/2},
                    {(x3Max - x3Min)/2, -(u1Max - u1Min)/2, -(u2Max - u2Min)/2},
                    {-(x3Max - x3Min)/2, (u1Max - u1Min)/2, (u2Max - u2Min)/2},
                    {-(x3Max - x3Min)/2, (u1Max - u1Min)/2, -(u2Max - u2Min)/2},
                    {-(x3Max - x3Min)/2, -(u1Max - u1Min)/2, (u2Max - u2Min)/2},
                    {-(x3Max - x3Min)/2, -(u1Max - u1Min)/2, -(u2Max - u2Min)/2}
                };
        
                randomDirections = Join[
                   defaultDirections,
                   Table[Normalize[RandomReal[{-1, 1}, 3]], {4}]
                   ];
                
                pointToList[point_] := {x3, u1, u2} /. point;
                listToPoint[list_] := Thread[{x3, u1, u2} -> list];
                
                findBoundaryPoint3D[pointRules_, direction_] := Module[
                    {
                        point = pointToList[pointRules], 
                        tLow = 0, 
                        tHigh = 20, 
                        maxIterations = 1000, 
                        iteration = 0, 
                        currentPoint
                    },
                    While[
                        RegionMember[boundedRegion, point + tHigh direction] && 
                         iteration < maxIterations,
        
                        tLow = tHigh;
                        tHigh *= 2;
                        iteration++;
                    ];
        
                    If[
                        iteration == maxIterations,
                        Return[listToPoint[point + tLow direction]]
                    ];
        
                    iteration = 0;
                    While[
                        Abs[tHigh - tLow] > 0.00001 && iteration < maxIterations,
        
                        currentPoint = point + ((tLow + tHigh)/2) direction;
                        If[
                            RegionMember[boundedRegion, currentPoint],
        
                            tLow = (tLow + tHigh)/2,
                            tHigh = (tLow + tHigh)/2
                        ];
                        iteration++;
                    ];
        
                    listToPoint[point + tLow direction]
                ];
        
        
                boundaryPoints = findBoundaryPoint3D[feasiblePoint, #] & /@ randomDirections;
                boundaryPointsNumerical = pointToList /@ boundaryPoints;
                innerConvexMesh = ConvexHullMesh[boundaryPointsNumerical];
        
                faces = MeshCells[innerConvexMesh, 2];
                vertices = MeshCoordinates[innerConvexMesh];
        
                planeEquation[faceIndices_List] := Module[
                    {v1, v2, v3, normal, d},
                    {v1, v2, v3} = vertices[[Take[faceIndices, 3]]];
                    normal = Normalize[Cross[v2 - v1, v3 - v1]];
                    d = normal . v1;
                    {normal, d}
                ];
        
                planeEquations = planeEquation /@ (faces[[All, 1]]);
        
                inequalities = 
                  Table[With[{normal = eq[[1]], offset = eq[[2]]}, 
                    normal . {u1, u2, x3} <= offset], {eq, planeEquations}];
        
                inequalities
                ];
                res
            '''))
            session.stop()

            # print('expressions', expressions)
            # for expr in expressions:
            #     print('expression', expr)
            #     print('sympy', _convert_wl_to_sympy(expr))

            return (
                [
                    Le(2, x1),
                    Le(x1, -2),  # make it infeasible
                ] if expressions is None or len(expressions) == 0 else
                [
                _convert_wl_to_sympy(expr) for expr in expressions
                ] + [
                    # Le(sMin, x1),
                    # Le(x1, sMax),
                    # Le(nMin, x2),
                    # Le(x2, nMax),
                    Le(vyMin, x4),
                    Le(x4, vyMax),
                ],
                [x1, x2, x3, x4, u1, u2]
            )
        except WolframKernelException as e:
            print(e)
            kill_wolfram_kernel()
            time.sleep(2)
            session = WolframLanguageSession('/Applications/Wolfram.app/Contents/MacOS/WolframKernel')

