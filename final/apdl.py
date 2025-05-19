import random
print("Script started: Imported random module.")

# --- Configuration ---
NUMBER_OF_RUNS = 1000
print("Configuration: NUMBER_OF_RUNS = {}".format(NUMBER_OF_RUNS))

MIN_ELEMENT_SIZE_VALUE = 0.0105 
print("Configuration: MIN_ELEMENT_SIZE_VALUE = {}".format(MIN_ELEMENT_SIZE_VALUE))

MAX_ELEMENT_SIZE_VALUE = 0.1  
print("Configuration: MAX_ELEMENT_SIZE_VALUE = {}".format(MAX_ELEMENT_SIZE_VALUE))

ELEMENT_SIZE_UNITS = "m" 
print("Configuration: ELEMENT_SIZE_UNITS = '{}'".format(ELEMENT_SIZE_UNITS))

APDL_COMMAND_OBJECT_NAME = "Commands (APDL)"
print("Configuration: APDL_COMMAND_OBJECT_NAME = '{}'".format(APDL_COMMAND_OBJECT_NAME))

ANALYSIS_NAME = "Transient Thermal"
print("Configuration: ANALYSIS_NAME = '{}'".format(ANALYSIS_NAME))

# --- Get Model ---
print("\nAttempting to get Model object from ExtAPI.DataModel.Project.Model...")
model = ExtAPI.DataModel.Project.Model
print("Model object obtained: {}".format(model))

# --- Find Analysis ---
print("\nFinding Analysis System '{}'...".format(ANALYSIS_NAME))
analysis = None
if model.Analyses:
    for ansys_analysis_item in model.Analyses: 
        if ansys_analysis_item.Name == ANALYSIS_NAME:
            analysis = ansys_analysis_item
            break
if analysis is None:
    print("CRITICAL ERROR: Analysis named '{}' was NOT found.".format(ANALYSIS_NAME))
    raise Exception("Analysis not found.")
print("Analysis '{}' found: {}".format(analysis.Name, analysis))

# --- Get Solution ---
print("\nGetting Solution object from analysis '{}'...".format(analysis.Name))
solution = analysis.Solution
print("Solution object obtained: {} (Name: {})".format(solution, solution.Name))

# --- Verify Mesh Object and Element Size Access ---
print("\nVerifying Mesh Object and ElementSize access using Quantity...")
if model.Mesh is None:
    print("CRITICAL ERROR: model.Mesh object is None.")
    raise Exception("Model.Mesh object is None.")
else:
    print("Model.Mesh object exists: {}".format(model.Mesh))
    try:
        if hasattr(model.Mesh, "ElementSize"):
            current_element_size_quantity = model.Mesh.ElementSize
            
            initial_unit_str = "N/A"
            if current_element_size_quantity.Unit: 
                if hasattr(current_element_size_quantity.Unit, "Name"):
                    initial_unit_str = current_element_size_quantity.Unit.Name
                elif isinstance(current_element_size_quantity.Unit, str): 
                    initial_unit_str = current_element_size_quantity.Unit
                else: 
                    initial_unit_str = str(current_element_size_quantity.Unit)
            
            print("  Initial model.Mesh.ElementSize: (Full: '{}', Value: {}, Unit String: '{}')".format(
                str(current_element_size_quantity), 
                current_element_size_quantity.Value, 
                initial_unit_str
            ))
        else:
            print("  CRITICAL ERROR: model.Mesh does not have 'ElementSize' property. Check API name.")
            raise AttributeError("model.Mesh object has no attribute 'ElementSize'")
        
        test_value_float = random.uniform(MIN_ELEMENT_SIZE_VALUE, MAX_ELEMENT_SIZE_VALUE) 
        test_element_size_quantity = Quantity(test_value_float, ELEMENT_SIZE_UNITS)
        print("  Attempting to set model.Mesh.ElementSize to: {}".format(test_element_size_quantity))
        
        model.Mesh.ElementSize = test_element_size_quantity
        
        new_element_size_quantity = model.Mesh.ElementSize
        
        new_unit_str = "N/A"
        if new_element_size_quantity.Unit: 
            if hasattr(new_element_size_quantity.Unit, "Name"):
                new_unit_str = new_element_size_quantity.Unit.Name
            elif isinstance(new_element_size_quantity.Unit, str): 
                new_unit_str = new_element_size_quantity.Unit
            else: 
                new_unit_str = str(new_element_size_quantity.Unit)

        print("  New model.Mesh.ElementSize after setting: (Full: '{}', Value: {}, Unit String: '{}')".format(
            str(new_element_size_quantity), 
            new_element_size_quantity.Value,
            new_unit_str
        ))

        if abs(new_element_size_quantity.Value - test_value_float) < 1e-9: 
            print("  SUCCESS: Mesh ElementSize changed and verified.")
        else:
            print("  WARNING: Mesh ElementSize numerical value set to {} but reads back as {}. This might be due to internal adjustments or floating point precision.".format(test_value_float, new_element_size_quantity.Value))

    except AttributeError as e_attr: 
        print("  ATTRIBUTE ERROR: {}".format(str(e_attr)))
        raise 
    except Exception as e_mesh_prop: 
        print("  UNEXPECTED ERROR accessing or setting model.Mesh.ElementSize: {}".format(str(e_mesh_prop)))
        if "Quantity" in str(e_mesh_prop) and "not defined" in str(e_mesh_prop):
             print("    HINT: 'Quantity' might not be defined. Ensure it's imported or accessed correctly for your Ansys version.")
        raise 
    print("  Mesh object and ElementSize property appear accessible and modifiable with Quantity.")

# --- Locate APDL Command Object ---
print("\nLocating APDL Command Object '{}'...".format(APDL_COMMAND_OBJECT_NAME))
apdl_command_obj = None
if solution.Children:
    for child in solution.Children:
        if child.Name == APDL_COMMAND_OBJECT_NAME:
            apdl_command_obj = child
            break 
if apdl_command_obj is None:
    print("CRITICAL ERROR: APDL Command object named '{}' NOT found under Solution '{}'.".format(APDL_COMMAND_OBJECT_NAME, solution.Name))
    raise Exception("APDL Command object not found.")
else:
    print("APDL Command object '{}' found: {}".format(apdl_command_obj.Name, apdl_command_obj))
    if not hasattr(apdl_command_obj, "Input"):
        print("CRITICAL ERROR: APDL object '{}' does not have an 'Input' property.".format(apdl_command_obj.Name))
        raise Exception("APDL object lacks 'Input' property.")
    print("  APDL object has '.Input' property. Current APDL text (first 70 chars): '{}...'".format(apdl_command_obj.Input[:70]))

# --- Main Automation Loop ---
smat_command_line = "*SMAT, MatKS, D, IMPORT, FULL, file.full, STIFF"
print("\nBase SMAT command line for APDL: '{}'".format(smat_command_line))
print("\n--- Starting Main Automation Loop ---")
ExtAPI.Log.WriteMessage("Script: Starting automation loop for {} runs.".format(NUMBER_OF_RUNS))

for i_run in range(1, NUMBER_OF_RUNS + 1):
    print("\n" + "=" * 40)
    print("Beginning Run {} of {}".format(i_run, NUMBER_OF_RUNS))
    ExtAPI.Log.WriteMessage("Script: Beginning Run {}/{}.".format(i_run, NUMBER_OF_RUNS))

    # 1. Change Mesh Element Size
    print("  Step 1: Change Mesh Element Size")
    target_value_float = random.uniform(MIN_ELEMENT_SIZE_VALUE, MAX_ELEMENT_SIZE_VALUE)
    target_element_size_quantity = Quantity(target_value_float, ELEMENT_SIZE_UNITS) 
    print("    Target element size for this run: {}".format(target_element_size_quantity))
    try:
        model.Mesh.ElementSize = target_element_size_quantity
        read_back_quantity = model.Mesh.ElementSize
        
        loop_read_back_unit_str = "N/A"
        if read_back_quantity.Unit:
            if hasattr(read_back_quantity.Unit, "Name"):
                loop_read_back_unit_str = read_back_quantity.Unit.Name
            elif isinstance(read_back_quantity.Unit, str):
                loop_read_back_unit_str = read_back_quantity.Unit
            else:
                loop_read_back_unit_str = str(read_back_quantity.Unit)
        
        print("    model.Mesh.ElementSize successfully set to: (Full: '{}', Value: {}, Unit String: '{}')".format(
            str(read_back_quantity),
            read_back_quantity.Value, 
            loop_read_back_unit_str
        ))
        ExtAPI.Log.WriteMessage("Script Run {}: Mesh ElementSize set to {}.".format(i_run, model.Mesh.ElementSize))
    except Exception as e_mesh:
        print("    ERROR setting mesh element size: {}. Skipping this run.".format(str(e_mesh)))
        if "Quantity" in str(e_mesh) and "not defined" in str(e_mesh):
             print("    HINT: 'Quantity' might not be defined. Ensure it's imported or accessed correctly for your Ansys version.")
        ExtAPI.Log.WriteMessage("Script Run {}: ERROR setting mesh element size: {}. Skipping run.".format(i_run, str(e_mesh)))
        continue 

    # 2. Modify APDL Command
    print("  Step 2: Modify APDL Command")
    output_filename = "Ksparse{}.matrix".format(i_run) 
    print("    Output filename for APDL: '{}'".format(output_filename))
    print_command_line = "*PRINT, MatKS, {}".format(output_filename)
    full_apdl_script = "{}\n{}".format(smat_command_line, print_command_line)
    print("    Full APDL script for this run:\n---\n{}\n---".format(full_apdl_script))
    try:
        apdl_command_obj.Input = full_apdl_script
        print("    APDL '.Input' property updated successfully.")
        ExtAPI.Log.WriteMessage("Script Run {}: APDL '.Input' property updated for file '{}'.".format(i_run, output_filename))
    except Exception as e_apdl:
        print("    ERROR updating APDL '.Input' property: {}. Skipping this run.".format(str(e_apdl)))
        ExtAPI.Log.WriteMessage("Script Run {}: ERROR updating APDL '.Input' property: {}. Skipping run.".format(i_run, str(e_apdl)))
        continue

    # 3. Solve the Analysis
    print("  Step 3: Solve the Analysis")
    ExtAPI.Log.WriteMessage("Script Run {}: Attempting to solve...".format(i_run))
    try:
        print("    Attempting solution.Solve(True)...")
        solution.Solve(True) 
        print("    Solve successful for run {}.".format(i_run))
        ExtAPI.Log.WriteMessage("Script Run {}: Solve successful.".format(i_run))
        ExtAPI.Log.WriteMessage("Script Run {}: Matrix file '{}' should be created.".format(i_run, output_filename))
    except Exception as e_solve:
        print("    ERROR during solve for run {}: {}".format(i_run, str(e_solve)))
        ExtAPI.Log.WriteMessage("Script Run {}: ERROR during solve: {}. Skipping run.".format(i_run, str(e_solve)))
        continue 
    
    print("  --- End of processing for Run {} ---".format(i_run))

print("\n" + "=" * 40)
ExtAPI.Log.WriteMessage("Script: Automation loop finished. All runs attempted.") 
print("Script execution finished. All runs attempted.")