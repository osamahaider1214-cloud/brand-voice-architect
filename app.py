def show_consistency_audit_tabs():
    audit_data = get_audit_data()
    tabs = st.tabs(['Overview', 'Detail', 'Consistency Audit'])

    with tabs[2]:  # Consistency Audit tab
        if "overall" in audit_data:
            st.divider()
            st.code(pretty_json(audit_data), language="json")

# Other code...
