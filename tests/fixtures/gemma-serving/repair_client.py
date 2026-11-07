"""Frozen serving prompt for mocked client tests; see README.md."""

SYSTEM = 'Propose a minimal correction to the failed hand-over plan. Return only a flat JSON object matching PlanPatch, without any wrapper key or explanation. Allowed keys: grasp_height_offset_m, approach_yaw_deg, pregrasp_clearance_m, lift_height_m, handover_point_m, handover_yaw_offset_deg, gripper_close_delay_s, release_delay_s, release_height_m, max_speed_m_s. Include only changed fields. MuJoCo will independently verify the proposed plan.'
