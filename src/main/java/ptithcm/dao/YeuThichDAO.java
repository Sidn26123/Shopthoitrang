package ptithcm.dao;

import java.util.List;

import ptithcm.entity.YeuThichEntity;

public interface YeuThichDAO {
	public List<YeuThichEntity> layDSYeuThichCuaUser(int maNd);
	public YeuThichEntity layYeuThichTheoMaNDVaSanPham(int maNd, String maSp);
	public void addYeuThich(YeuThichEntity yeuThich);
	public void deleteYeuThich(int maYT);
}
