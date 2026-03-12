import time
from scripts.gripper.susgrip_client import SusGripClient


def print_status(tag, s):
    print(
        f"{tag}: POS={s.pos_mm}mm  SPD={s.speed_mm_s}  MOT={s.motion}  "
        f"ERR={s.error}  OBJ=0x{s.obj:04X}  EMCY=0x{s.emcy:04X}  "
        f"V={s.vbus_mv}mV  I={s.ibus_ma}mA"
    )


def main():
    g = SusGripClient(
        port="COM4",  # change if your COM changed
        slave_id=1,
        open_max_mm=120,
        close_min_mm=10,
    )

    print("Connecting...")
    if not g.connect():
        print("❌ Could not connect. (GUI still open? wrong COM?)")
        return

    try:
        # Always start safe
        g.emcy_release()
        g.set_speed_force(speed_mm_s=20, force_pct=10)

        s = g.get_status()
        print_status("START", s)

        print("\nOpening to max...")
        ok = g.open(wait=True)
        s = g.get_status()
        print_status("AFTER OPEN", s)
        print(f"open() finished: {ok}")

        time.sleep(0.5)

        print("\nClosing to min...")
        ok = g.close(wait=True)
        s = g.get_status()
        print_status("AFTER CLOSE", s)
        print(f"close() finished: {ok}")

    except KeyboardInterrupt:
        print("\n⚠️ Ctrl+C pressed → EMCY STOP")
        try:
            g.emcy_stop()
        except Exception:
            pass
    finally:
        g.close()
        print("\nDisconnected (COM released).")


if __name__ == "__main__":
    main()
