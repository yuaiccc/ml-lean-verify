import PolyLeanVerify
import Mathlib.Tactic
open PolyLeanVerify

/-- Parse a comma-separated list of floats from a string -/
def parseFloatList (s : String) : List Float :=
  s.splitOn "," |>.filterMap (fun part =>
    let trimmed := part.trimAscii.toString
    if trimmed.isEmpty then none
    else some (stringToFloat trimmed))
where
  stringToFloat (s : String) : Float :=
    if s.startsWith "-" then
      -1.0 * stringToFloatAbs (s.drop 1 |>.toString)
    else
      stringToFloatAbs s

  stringToFloatAbs (s : String) : Float :=
    match s.splitOn "." with
    | [intPart] => if intPart.isEmpty then 0.0 else intPart.toNat!.toFloat
    | [intPart, fracPart] =>
      let intVal := if intPart.isEmpty then 0.0 else intPart.toNat!.toFloat
      let divisor := (10 ^ fracPart.length : Nat).toFloat
      let fracVal := if fracPart.isEmpty then 0.0 else fracPart.toNat!.toFloat / divisor
      intVal + fracVal
    | _ => 0.0

def main : IO Unit := do
  let stdin ← IO.getStdin
  let line ← stdin.getLine
  let parts := line.trimAscii.toString.splitOn "|"

  if parts.length < 1 then
    IO.println "ERROR: no command"
    return

  let cmd := parts[0]!.trimAscii.toString

  match cmd with
  | "softmax" =>
    -- Format: softmax|x1,x2,x3,...
    if parts.length < 2 then
      IO.println "ERROR: softmax needs data"
      return
    let xs := parseFloatList parts[1]!
    let result := softmax xs
    -- Output as comma-separated
    let strs := result.map (fun x => Float.toString x)
    IO.println s!"SOFTMAX|{String.intercalate "," strs}"

  | "rmsnorm" =>
    -- Format: rmsnorm|x1,x2,x3,...|eps
    if parts.length < 3 then
      IO.println "ERROR: rmsnorm needs data and eps"
      return
    let xs := parseFloatList parts[1]!
    let eps := match parts[2]!.trimAscii.toString.splitOn "." with
      | [ip, fp] =>
        let iv := if ip.isEmpty then 0.0 else ip.toNat!.toFloat
        let dv := (10 ^ fp.length : Nat).toFloat
        let fv := if fp.isEmpty then 0.0 else fp.toNat!.toFloat / dv
        iv + fv
      | [ip] => if ip.isEmpty then 0.0 else ip.toNat!.toFloat
      | _ => 1e-5
    let result := rmsNorm xs eps
    let strs := result.map (fun x => Float.toString x)
    IO.println s!"RMSNORM|{String.intercalate "," strs}"

  | "layernorm" =>
    -- Format: layernorm|x1,x2,x3,...|eps
    if parts.length < 3 then
      IO.println "ERROR: layernorm needs data and eps"
      return
    let xs := parseFloatList parts[1]!
    let eps := match parts[2]!.trimAscii.toString.splitOn "." with
      | [ip, fp] =>
        let iv := if ip.isEmpty then 0.0 else ip.toNat!.toFloat
        let dv := (10 ^ fp.length : Nat).toFloat
        let fv := if fp.isEmpty then 0.0 else fp.toNat!.toFloat / dv
        iv + fv
      | [ip] => if ip.isEmpty then 0.0 else ip.toNat!.toFloat
      | _ => 1e-5
    let result := layerNorm xs eps
    let strs := result.map (fun x => Float.toString x)
    IO.println s!"LAYERNORM|{String.intercalate "," strs}"

  | "rope" =>
    -- Format: rope|x1,x2,x3,...|pos|base
    if parts.length < 4 then
      IO.println "ERROR: rope needs data, pos, base"
      return
    let xs := parseFloatList parts[1]!
    let pos := match parts[2]!.trimAscii.toString.splitOn "." with
      | [ip, fp] =>
        let iv := if ip.isEmpty then 0.0 else ip.toNat!.toFloat
        let dv := (10 ^ fp.length : Nat).toFloat
        let fv := if fp.isEmpty then 0.0 else fp.toNat!.toFloat / dv
        iv + fv
      | [ip] => if ip.isEmpty then 0.0 else ip.toNat!.toFloat
      | _ => 0.0
    let base := match parts[3]!.trimAscii.toString.splitOn "." with
      | [ip, fp] =>
        let iv := if ip.isEmpty then 0.0 else ip.toNat!.toFloat
        let dv := (10 ^ fp.length : Nat).toFloat
        let fv := if fp.isEmpty then 0.0 else fp.toNat!.toFloat / dv
        iv + fv
      | [ip] => if ip.isEmpty then 0.0 else ip.toNat!.toFloat
      | _ => 10000.0
    let result := ropeApply xs pos base
    let strs := result.map (fun x => Float.toString x)
    IO.println s!"ROPE|{String.intercalate "," strs}"

  | "activeFraction" =>
    -- Format: activeFraction|variant|nExperts|topK|nShared
    if parts.length < 5 then
      IO.println "ERROR: activeFraction needs variant, nExperts, topK, nShared"
      return
    let variantStr := parts[1]!.trimAscii.toString
    let nExperts := parts[2]!.trimAscii.toString.toNat!
    let topK := parts[3]!.trimAscii.toString.toNat!
    let nShared := parts[4]!.trimAscii.toString.toNat!
    let variant := match variantStr with
      | "dense" => MoEVariant.dense
      | "moe" => MoEVariant.moe
      | "switch" => MoEVariant.switch
      | "deepseek" => MoEVariant.deepseek
      | _ => MoEVariant.moe
    let result := activeFraction variant nExperts topK nShared
    IO.println s!"ACTIVEFRACTION|{Float.toString result}"

  | "gradMagnitude" =>
    -- Format: gradMagnitude|variant|depth|nLayers|w|streams
    if parts.length < 6 then
      IO.println "ERROR: gradMagnitude needs variant, depth, nLayers, w, streams"
      return
    let variantStr := parts[1]!.trimAscii.toString
    let depth := parts[2]!.trimAscii.toString.toNat!
    let nLayers := parts[3]!.trimAscii.toString.toNat!
    let w := match parts[4]!.trimAscii.toString.splitOn "." with
      | [ip, fp] =>
        let iv := if ip.isEmpty then 0.0 else ip.toNat!.toFloat
        let dv := (10 ^ fp.length : Nat).toFloat
        let fv := if fp.isEmpty then 0.0 else fp.toNat!.toFloat / dv
        iv + fv
      | [ip] => if ip.isEmpty then 0.0 else ip.toNat!.toFloat
      | _ => 0.8
    let streams := parts[5]!.trimAscii.toString.toNat!
    let variant := match variantStr with
      | "plain" => ResidualVariant.plain
      | "rc" => ResidualVariant.rc
      | "hc" => ResidualVariant.hc
      | "mhc" => ResidualVariant.mhc
      | "attnres" => ResidualVariant.attnres
      | _ => ResidualVariant.rc
    let result := gradMagnitude variant depth nLayers w streams
    IO.println s!"GRADMAG|{Float.toString result}"

  | _ => IO.println "ERROR: unknown command"
