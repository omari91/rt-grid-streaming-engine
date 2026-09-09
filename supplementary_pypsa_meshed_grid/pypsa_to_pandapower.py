import pypsa
import pandapower as pp
import numpy as np

def convert_pypsa_to_pandapower(n: pypsa.Network) -> pp.pandapowerNet:
    """
    Converts a PyPSA network to a Pandapower network.
    Extracts buses, lines, loads, and generators.
    """
    net = pp.create_empty_network()

    # Create buses
    bus_map = {}
    for bus_name, bus in n.buses.iterrows():
        bus_map[bus_name] = pp.create_bus(
            net,
            vn_kv=20.0,
            name=str(bus_name),
            geodata=(bus.get('x', 0.0), bus.get('y', 0.0))
        )

    # Create lines
    # For PyPSA lines, standard types might not be present, so we use custom line parameters
    for line_name, line in n.lines.iterrows():
        bus0 = bus_map.get(line['bus0'])
        bus1 = bus_map.get(line['bus1'])
        if bus0 is None or bus1 is None: continue
        
        # PyPSA line parameters are per unit or physical. 
        # r, x, b, g are in ohms, siemens etc if length > 0.
        length = line.get('length', 1.0)
        if length <= 0.0: length = 1.0
            
        r_ohm_per_km = line.get('r', 0.0) / length if 'r' in line else 0.3
        x_ohm_per_km = line.get('x', 0.0) / length if 'x' in line else 0.1
        c_nf_per_km = 200.0  # typical for 20kV cable
        max_i_ka = line.get('s_nom', 1000.0) / (np.sqrt(3) * net.bus.at[bus0, 'vn_kv']) if 's_nom' in line else 1.0

        pp.create_line_from_parameters(
            net, 
            bus0, 
            bus1, 
            length_km=length,
            r_ohm_per_km=r_ohm_per_km, 
            x_ohm_per_km=x_ohm_per_km,
            c_nf_per_km=c_nf_per_km,
            max_i_ka=max_i_ka,
            name=str(line_name)
        )

    # Create loads
    for load_name, load in n.loads.iterrows():
        bus = bus_map.get(load['bus'])
        if bus is None: continue
        p_mw = load.get('p_set', 0.0)
        q_mvar = load.get('q_set', 0.0)
        pp.create_load(net, bus, p_mw=p_mw, q_mvar=q_mvar, name=str(load_name))

    # Create generators
    # In PyPSA, Slack bus is defined by generator control='Slack'
    for gen_name, gen in n.generators.iterrows():
        bus = bus_map.get(gen['bus'])
        if bus is None: continue
        p_mw = gen.get('p_set', gen.get('p_nom', 0.0))
        
        control = gen.get('control', '')
        if control == 'Slack':
            pp.create_ext_grid(net, bus, vm_pu=1.0, name=str(gen_name))
        else:
            pp.create_sgen(net, bus, p_mw=p_mw, q_mvar=0.0, name=str(gen_name))
            
    # Ensure at least one slack bus exists
    if len(net.ext_grid) == 0 and len(net.bus) > 0:
        first_bus = net.bus.index[0]
        pp.create_ext_grid(net, first_bus, vm_pu=1.0, name="Fallback_Slack")

    return net

if __name__ == "__main__":
    print("Generating synthetic PyPSA transmission subgrid...")
    
    n = pypsa.Network()
    
    # Create 15 buses (380kV)
    for i in range(15):
        n.add("Bus", f"Bus_{i}", v_nom=380.0)
        
    # Create some lines (MV topology)
    for i in range(14):
        n.add("Line", f"Line_{i}", bus0=f"Bus_{i}", bus1=f"Bus_{i+1}", length=15.0, x=0.2, r=0.6, s_nom=20.0)
    # Add a loop
    n.add("Line", "Line_loop", bus0="Bus_0", bus1="Bus_14", length=25.0, x=0.4, r=1.2, s_nom=20.0)
    
    # Add some generators
    n.add("Generator", "Gen_Slack", bus="Bus_0", control="Slack", p_nom=200.0)
    n.add("Generator", "Gen_Wind1", bus="Bus_5", p_nom=20.0, p_set=10.0)
    n.add("Generator", "Gen_Solar1", bus="Bus_10", p_nom=15.0, p_set=8.0)
    
    # Add some loads
    for i in range(1, 15):
        n.add("Load", f"Load_{i}", bus=f"Bus_{i}", p_set=1.0 + (i * 0.1))
        
    print(n)
    
    print("Converting to Pandapower...")
    net = convert_pypsa_to_pandapower(n)
    print(f"Converted to Pandapower: \n{net}")
    
    import os
    os.makedirs("data", exist_ok=True)
    pp.to_json(net, "data/pypsa_de_subgrid.json")
    print("Saved to data/pypsa_de_subgrid.json")
